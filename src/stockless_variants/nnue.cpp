// Stockless Variants contributors, 2026. SPDX-License-Identifier: AGPL-3.0-only
// Independent, bounded, quantized SVNNUE-v1 inference and transactional loader.
#include "nnue.h"
#include "nnue_rules.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

#include "../position.h"
#include "../variant.h"

namespace Stockfish::Stockless::Neural {
namespace {

constexpr const char* NativeVariants[] = {"chess", "shogi", "janggi", "xiangqi", "makruk"};
constexpr std::size_t MaximumFileBytes = 4 * 1024 * 1024;
constexpr std::int32_t MaximumBias = 1 << 20;
constexpr std::uint8_t Magic[8] = {'S', 'V', 'N', 'N', 'U', 'E', '1', 0};
constexpr std::uint8_t MagicV2[8] = {'S', 'V', 'N', 'N', 'U', 'E', '2', 0};

enum class Backend { Classical, Auto, Neural };

struct Descriptor {
    std::string name;
    int width = 0;
    int height = 0;
    int area = 0;
    std::vector<int> types;
    std::array<int, PIECE_TYPE_NB> type_index;
    std::uint32_t base_dimensions = 0;
    std::uint32_t dimensions = 0;
    bool v2 = false;
};

struct Model {
    const Variant* variant = nullptr;
    Descriptor descriptor;
    std::vector<std::int16_t> input_weights;
    std::array<std::int32_t, Hidden> input_bias{};
    std::array<std::array<std::int16_t, Hidden>, COLOR_NB> output_weights{};
    std::array<std::int32_t, COLOR_NB> output_bias{};
    std::string file;
};

std::unique_ptr<Model> model;
Backend selected_backend = Backend::Classical;
std::uint64_t network_generation = 1;

static_assert(std::is_trivially_copyable<Accumulator>::value, "Position requires a POD accumulator");

void advance_generation() {
    if (++network_generation == 0)
        ++network_generation;
}

std::string native_name(const Variant* variant) {
    if (!variant)
        return {};
    for (const char* name : NativeVariants) {
        const auto found = variants.find(name);
        if (found != variants.end() && found->second == variant)
            return name;
    }
    return {};
}

Descriptor describe(const Variant* variant, bool v2 = false) {
    Descriptor result;
    result.v2 = v2;
    result.type_index.fill(-1);
    result.name = native_name(variant);
    if (result.name.empty())
        return result;
    result.width = int(variant->maxFile) + 1;
    result.height = int(variant->maxRank) + 1;
    result.area = result.width * result.height;
    for (int pt = int(PAWN); pt <= int(KING); ++pt)
        if (variant->pieceTypes & PieceType(pt)) {
            result.type_index[pt] = int(result.types.size());
            result.types.push_back(pt);
        }
    const int typeCount = int(result.types.size());
    result.base_dimensions = std::uint32_t(2 * typeCount * (result.area + 1));
    // V2 includes independent promoted-origin board features, chess
    // castling/en-passant rights and three per-piece regional buckets.
    result.dimensions = result.base_dimensions
       + (v2 ? std::uint32_t(2 * typeCount * result.area + 4 + result.width + 6 * typeCount) : 0);
    return result;
}

std::string quote(const std::string& text) {
    static const char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += char(c);
        } else if (c < 32) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else
            result += char(c);
    }
    result += '"';
    return result;
}

template<typename T, std::size_t N>
void json_array(std::ostream& stream, const std::array<T, N>& values) {
    stream << '[';
    for (std::size_t i = 0; i < N; ++i) {
        if (i) stream << ',';
        stream << values[i];
    }
    stream << ']';
}

std::uint32_t read_u32(const std::uint8_t* bytes) {
    return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8)
         | (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
}

std::int32_t read_i32(const std::uint8_t* bytes) {
    const std::uint32_t value = read_u32(bytes);
    return value <= std::uint32_t(std::numeric_limits<std::int32_t>::max())
         ? std::int32_t(value) : std::int32_t(std::int64_t(value) - INT64_C(4294967296));
}

std::int16_t read_i16(const std::uint8_t* bytes) {
    const std::uint32_t value = std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8);
    return std::int16_t(value <= 32767 ? std::int32_t(value) : std::int32_t(value) - 65536);
}

// SHA-256 over the exact payload bytes. Explicit byte order makes files
// independent of compiler struct padding and host endianness.
constexpr std::uint32_t ShaConstants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

std::uint32_t rotate_right(std::uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32 - bits));
}

void sha_block(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) {
    std::uint32_t words[64];
    for (int i = 0; i < 16; ++i)
        words[i] = (std::uint32_t(block[4 * i]) << 24) | (std::uint32_t(block[4 * i + 1]) << 16)
                 | (std::uint32_t(block[4 * i + 2]) << 8) | std::uint32_t(block[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
        const auto x = words[i - 15], y = words[i - 2];
        const auto s0 = rotate_right(x, 7) ^ rotate_right(x, 18) ^ (x >> 3);
        const auto s1 = rotate_right(y, 17) ^ rotate_right(y, 19) ^ (y >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    auto a = state[0], b = state[1], c = state[2], d = state[3];
    auto e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        const auto s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const auto choice = (e & f) ^ (~e & g);
        const auto t1 = h + s1 + choice + ShaConstants[i] + words[i];
        const auto s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const auto majority = (a & b) ^ (a & c) ^ (b & c);
        const auto t2 = s0 + majority;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

std::array<std::uint8_t, 32> sha256(const std::uint8_t* bytes, std::size_t length) {
    std::array<std::uint32_t, 8> state = {{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                         0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u}};
    std::size_t offset = 0;
    while (length - offset >= 64) {
        sha_block(state, bytes + offset);
        offset += 64;
    }
    std::uint8_t tail[128]{};
    const std::size_t remaining = length - offset;
    std::copy(bytes + offset, bytes + length, tail);
    tail[remaining] = 0x80;
    const std::size_t tail_length = remaining < 56 ? 64 : 128;
    const std::uint64_t bit_length = std::uint64_t(length) * 8;
    for (unsigned i = 0; i < 8; ++i)
        tail[tail_length - 1 - i] = std::uint8_t(bit_length >> (8 * i));
    sha_block(state, tail);
    if (tail_length == 128)
        sha_block(state, tail + 64);
    std::array<std::uint8_t, 32> result{};
    for (std::size_t i = 0; i < state.size(); ++i)
        for (unsigned j = 0; j < 4; ++j)
            result[4 * i + j] = std::uint8_t(state[i] >> (24 - 8 * j));
    return result;
}

bool rules_digest(std::array<std::uint8_t, 32>& result) {
    const std::string text = STOCKLESS_NNUE_RULES_SHA256;
    if (text.size() != 64)
        return false;
    auto digit = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (std::size_t i = 0; i < result.size(); ++i) {
        const int high = digit(text[2 * i]), low = digit(text[2 * i + 1]);
        if (high < 0 || low < 0)
            return false;
        result[i] = std::uint8_t((high << 4) | low);
    }
    return true;
}

bool compatible_variant(const Variant* variant) {
    return model && model->variant == variant && !native_name(variant).empty();
}

bool compatible_position(const Position& pos) {
    return compatible_variant(pos.variant()) && !pos.is_chess960();
}

bool enabled() {
    return selected_backend != Backend::Classical && bool(model);
}

int feature_index(const Descriptor& descriptor, Piece pc, int slot) {
    const int pt = int(type_of(pc));
    if (pc == NO_PIECE || unsigned(pc) >= PIECE_NB || pt >= PIECE_TYPE_NB
        || descriptor.type_index[pt] < 0 || slot < 0 || slot > descriptor.area)
        return -1;
    return (int(color_of(pc)) * int(descriptor.types.size()) + descriptor.type_index[pt])
           * (descriptor.area + 1) + slot;
}

bool extract(const Position& pos, const Descriptor& descriptor,
             std::vector<std::pair<std::uint32_t, std::int32_t>>& features,
             std::int32_t& mass) {
    mass = 0;
    if (descriptor.name.empty() || pos.is_chess960())
        return false;
    for (int rank = 0; rank < descriptor.height; ++rank)
        for (int file = 0; file < descriptor.width; ++file) {
            const Piece pc = pos.piece_on(make_square(File(file), Rank(rank)));
            if (pc == NO_PIECE)
                continue;
            const int index = feature_index(descriptor, pc, rank * descriptor.width + file);
            if (index < 0)
                return false;
            features.emplace_back(std::uint32_t(index), 1);
            ++mass;
        }
    for (Color color : {WHITE, BLACK})
        for (int pt : descriptor.types) {
            const int count = pos.count_in_hand(color, PieceType(pt));
            if (count < 0 || count > MaximumFeatureMass || mass > MaximumFeatureMass - count)
                return false;
            if (count) {
                features.emplace_back(std::uint32_t(feature_index(descriptor, make_piece(color, PieceType(pt)), descriptor.area)), count);
                mass += count;
            }
        }
    if (mass > MaximumFeatureMass)
        return false;
    std::sort(features.begin(), features.end());
    return true;
}


// SVNNUE-v2 adds a separate, dynamically computed transformer contribution.
// The board/pocket accumulator remains incremental. Extra state (promoted
// origins and castling/en-passant rights) must NEVER be updated by ordinary
// put_piece()/move_piece() deltas; it is recomputed from the actual Position
// before each neural evaluation, which also avoids stale make/unmake state.
bool append_extra_features(const Position& pos, const Descriptor& descriptor,
                           std::vector<std::pair<std::uint32_t, std::int32_t>>& extra,
                           std::int32_t& mass) {
    if (!descriptor.v2)
        return true;
    const int count = int(descriptor.types.size());
    const std::uint32_t promoteBase = descriptor.base_dimensions;
    const std::uint32_t castleBase = promoteBase + std::uint32_t(2 * count * descriptor.area);
    const std::uint32_t epBase = castleBase + 4;
    const std::uint32_t regionBase = epBase + std::uint32_t(descriptor.width);
    std::vector<int> regional(6 * count, 0);
    const bool palaceGame = descriptor.name == "janggi" || descriptor.name == "xiangqi";
    const bool riverGame = descriptor.name == "xiangqi";
    const bool promotionGame = descriptor.name == "shogi" || descriptor.name == "makruk";
    for (int rank = 0; rank < descriptor.height; ++rank)
        for (int file = 0; file < descriptor.width; ++file) {
            const Square sq = make_square(File(file), Rank(rank));
            const Piece piece = pos.piece_on(sq);
            if (piece == NO_PIECE)
                continue;
            const int color = int(color_of(piece)), kind = int(type_of(piece));
            const int type = kind < PIECE_TYPE_NB ? descriptor.type_index[kind] : -1;
            if (color < 0 || color >= 2 || type < 0)
                return false;
            if (pos.is_promoted(sq)) {
                const Piece nativeOrigin = pos.unpromoted_piece_on(sq);
                const int originalType = int(type_of(nativeOrigin));
                const int oldIndex = nativeOrigin != NO_PIECE && originalType < PIECE_TYPE_NB
                                   ? descriptor.type_index[originalType] : -1;
                if (oldIndex >= 0) {
                    const auto index = promoteBase
                        + std::uint32_t((color * count + oldIndex) * descriptor.area
                                        + rank * descriptor.width + file);
                    extra.emplace_back(index, 1);
                    ++mass;
                }
            }
            const int group = color * count + type;
            if (palaceGame && file >= 3 && file <= 5
                    && (rank <= 2 || rank >= descriptor.height - 3))
                ++regional[group];
            if (riverGame && (color == int(WHITE) ? rank >= 5 : rank <= 4))
                ++regional[2 * count + group];
            if (promotionGame && (color == int(WHITE)
                                      ? rank >= descriptor.height - 3 : rank <= 2))
                ++regional[4 * count + group];
        }
    if (descriptor.name == "chess") {
        std::istringstream fen(pos.fen());
        std::string board, stm, castle, ep;
        if (!(fen >> board >> stm >> castle >> ep))
            return false;
        for (int i = 0; i < 4; ++i) {
            const char bit = "KQkq"[i];
            if (castle.find(bit) != std::string::npos) {
                extra.emplace_back(castleBase + std::uint32_t(i), 1);
                ++mass;
            }
        }
        if (ep != "-" && !ep.empty()) {
            const int file = int(ep[0]) - int('a');
            if (file < 0 || file >= descriptor.width)
                return false;
            extra.emplace_back(epBase + std::uint32_t(file), 1);
            ++mass;
        }
    }
    for (int i = 0; i < int(regional.size()); ++i) {
        if (regional[i] > 0) {
            extra.emplace_back(regionBase + std::uint32_t(i), regional[i]);
            mass += regional[i];
        }
    }
    if (mass > MaximumFeatureMass)
        return false;
    return true;
}

void add_weights(std::array<std::int32_t, Hidden>& accumulator,
                 const std::int16_t* weights, std::int32_t amount) {
#if defined(USE_AVX2)
    const __m256i multiplier = _mm256_set1_epi32(amount);
    for (unsigned i = 0; i < Hidden; i += 8) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + i));
        const __m256i expanded = _mm256_cvtepi16_epi32(packed);
        const __m256i previous = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator.data() + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulator.data() + i),
                           _mm256_add_epi32(previous, _mm256_mullo_epi32(expanded, multiplier)));
    }
#else
    for (unsigned i = 0; i < Hidden; ++i)
        accumulator[i] += std::int32_t(weights[i]) * amount;
#endif
}

bool refresh(const Position& pos, Accumulator& accumulator) {
    std::vector<std::pair<std::uint32_t, std::int32_t>> features;
    std::int32_t mass = 0;
    if (!model || !extract(pos, model->descriptor, features, mass)) {
        accumulator.generation = 0;
        return false;
    }
    accumulator.values = model->input_bias;
    accumulator.feature_mass = mass;
    for (const auto& feature : features)
        add_weights(accumulator.values, model->input_weights.data() + std::size_t(feature.first) * Hidden,
                    feature.second);
    accumulator.generation = network_generation;
    return true;
}

bool ensure_accumulator(const Position& pos) {
    auto& accumulator = pos.stockless_nnue_accumulator();
    if (!(accumulator.generation == network_generation || refresh(pos, accumulator)))
        return false;
    if (model && model->descriptor.v2) {
        std::vector<std::pair<std::uint32_t, std::int32_t>> extra;
        std::int32_t mass = accumulator.feature_mass;
        return append_extra_features(pos, model->descriptor, extra, mass);
    }
    return true;
}

std::int64_t numerator(const Position& pos, const Accumulator& accumulator) {
    const auto head = pos.side_to_move();
    std::array<std::int32_t, Hidden> combined = accumulator.values;
    if (model->descriptor.v2) {
        std::vector<std::pair<std::uint32_t, std::int32_t>> extra;
        std::int32_t mass = accumulator.feature_mass;
        if (!append_extra_features(pos, model->descriptor, extra, mass))
            return 0; // applicable() already rejected this state
        for (const auto& feature : extra)
            add_weights(combined, model->input_weights.data()
                        + std::size_t(feature.first) * Hidden, feature.second);
    }
    std::int64_t result = model->output_bias[head];
    for (unsigned i = 0; i < Hidden; ++i)
        result += std::int64_t(model->output_weights[head][i])
                * std::clamp(combined[i], std::int32_t(0), std::int32_t(QA));
    return result;
}

std::int32_t white_cp(std::int64_t sum) {
    constexpr std::int64_t divisor = std::int64_t(QA) * QB;
    const std::int64_t rounded = sum >= 0 ? (sum + divisor / 2) / divisor
                                          : -((-sum + divisor / 2) / divisor);
    return std::int32_t(std::clamp(rounded, -std::int64_t(MaximumWhiteCp), std::int64_t(MaximumWhiteCp)));
}

Value engine_value(const Position& pos, std::int32_t white) {
    const int scaled = white * int(PawnValueEg) / 100;
    return Value(pos.side_to_move() == WHITE ? scaled : -scaled);
}

void delta(const Position& pos, Piece pc, int slot, int amount) {
    auto& accumulator = pos.stockless_nnue_accumulator();
    if (!accumulator.generation)
        return;
    if (!enabled() || accumulator.generation != network_generation || !compatible_position(pos)) {
        accumulator.generation = 0;
        return;
    }
    const int index = feature_index(model->descriptor, pc, slot);
    const int mass = accumulator.feature_mass + amount;
    if (index < 0 || mass < 0 || mass > MaximumFeatureMass) {
        accumulator.generation = 0;
        return;
    }
    add_weights(accumulator.values, model->input_weights.data() + std::size_t(index) * Hidden, amount);
    accumulator.feature_mass = mass;
}

} // namespace

bool supported_variant(const Variant* variant) {
    return !native_name(variant).empty();
}

std::string variant_name(const Variant* variant) {
    auto name = native_name(variant);
    if (!name.empty())
        return name;
    for (const auto& item : variants)
        if (item.second == variant)
            return item.first;
    return {};
}

std::string descriptor_json(const Variant* variant, bool v2) {
    const auto descriptor = describe(variant, v2);
    std::ostringstream stream;
    stream << "{\"supported\":" << (descriptor.name.empty() ? "false" : "true")
           << ",\"format_version\":" << (v2 ? V2FormatVersion : FormatVersion)
           << ",\"feature_version\":" << (v2 ? V2FeatureVersion : FeatureVersion)
           << ",\"variant\":" << quote(variant_name(variant))
           << ",\"rules_sha256\":" << quote(STOCKLESS_NNUE_RULES_SHA256)
           << ",\"width\":" << descriptor.width << ",\"height\":" << descriptor.height
           << ",\"type_ids\":[";
    for (std::size_t i = 0; i < descriptor.types.size(); ++i) {
        if (i) stream << ',';
        stream << descriptor.types[i];
    }
    stream << "],\"hidden_dim\":" << Hidden << ",\"input_dim\":" << descriptor.dimensions
           << ",\"qa\":" << QA << ",\"qb\":" << QB;
    if (descriptor.name.empty())
        stream << ",\"error\":\"SVNNUE-v1 supports only native chess, shogi, janggi, xiangqi and makruk\"";
    stream << '}';
    return stream.str();
}

std::vector<std::pair<std::uint32_t, std::int32_t>> sparse_features(const Position& pos, bool v2) {
    std::vector<std::pair<std::uint32_t, std::int32_t>> result;
    std::int32_t mass = 0;
    const auto descriptor = describe(pos.variant(), v2);
    if (!extract(pos, descriptor, result, mass)
            || (v2 && !append_extra_features(pos, descriptor, result, mass)))
        result.clear();
    else
        std::sort(result.begin(), result.end());
    return result;
}

bool load_network(const std::string& path, const Variant* variant, std::string& error) {
    error.clear();
    auto descriptor = describe(variant);
    if (descriptor.name.empty()) {
        error = "SVNNUE-v1/v2 does not support this variant";
        return false;
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "unable to open SVNNUE network file";
        return false;
    }
    const auto length = input.tellg();
    if (length < std::streamoff(HeaderBytes) || length > std::streamoff(MaximumFileBytes)) {
        error = "invalid SVNNUE file size";
        return false;
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()))) {
        error = "truncated SVNNUE network file";
        return false;
    }
    const bool oldFormat = std::equal(std::begin(Magic), std::end(Magic), bytes.begin());
    const bool newFormat = std::equal(std::begin(MagicV2), std::end(MagicV2), bytes.begin());
    if (!oldFormat && !newFormat) {
        error = "unsupported network magic; expected independent SVNNUE-v1/v2";
        return false;
    }
    descriptor = describe(variant, newFormat);
    const std::uint32_t expected[] = {newFormat ? V2FormatVersion : FormatVersion,
                                      newFormat ? V2FeatureVersion : FeatureVersion,
                                      Hidden, descriptor.dimensions,
                                     std::uint32_t(descriptor.width), std::uint32_t(descriptor.height),
                                     std::uint32_t(descriptor.types.size()), QA, QB};
    for (unsigned i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i)
        if (read_u32(bytes.data() + 8 + 4 * i) != expected[i]) {
            error = "incompatible SVNNUE version, feature layout, geometry or quantization";
            return false;
        }
    const std::size_t payload_size = std::size_t(descriptor.dimensions) * Hidden * 2
                                 + Hidden * 4 + COLOR_NB * Hidden * 2 + COLOR_NB * 4;
    if (read_u32(bytes.data() + 44) != payload_size || bytes.size() != HeaderBytes + payload_size) {
        error = "SVNNUE payload size mismatch or trailing data";
        return false;
    }
    std::array<std::uint8_t, 32> expected_name{};
    std::copy(descriptor.name.begin(), descriptor.name.end(), expected_name.begin());
    if (!std::equal(expected_name.begin(), expected_name.end(), bytes.begin() + 48)) {
        error = "SVNNUE network belongs to a different variant";
        return false;
    }
    std::array<std::uint8_t, 32> digest{};
    if (!rules_digest(digest) || !std::equal(digest.begin(), digest.end(), bytes.begin() + 80)) {
        error = "SVNNUE rule/feature source fingerprint mismatch";
        return false;
    }
    if (std::any_of(bytes.begin() + 144, bytes.begin() + HeaderBytes, [](std::uint8_t byte) { return byte != 0; })) {
        error = "unsupported nonzero SVNNUE reserved header fields";
        return false;
    }
    digest = sha256(bytes.data() + HeaderBytes, payload_size);
    if (!std::equal(digest.begin(), digest.end(), bytes.begin() + 112)) {
        error = "SVNNUE payload SHA256 mismatch";
        return false;
    }
    auto candidate = std::make_unique<Model>();
    candidate->variant = variant;
    candidate->descriptor = descriptor;
    candidate->file = path;
    candidate->input_weights.resize(std::size_t(descriptor.dimensions) * Hidden);
    std::size_t offset = HeaderBytes;
    for (auto& weight : candidate->input_weights) {
        weight = read_i16(bytes.data() + offset);
        offset += 2;
    }
    for (auto& bias : candidate->input_bias) {
        bias = read_i32(bytes.data() + offset);
        offset += 4;
        if (bias < -MaximumBias || bias > MaximumBias) {
            error = "SVNNUE input bias exceeds safe accumulator bounds";
            return false;
        }
    }
    for (auto& head : candidate->output_weights)
        for (auto& weight : head) {
            weight = read_i16(bytes.data() + offset);
            offset += 2;
        }
    for (auto& bias : candidate->output_bias) {
        bias = read_i32(bytes.data() + offset);
        offset += 4;
    }
    model = std::move(candidate);
    advance_generation();
    return true;
}

bool select_backend(const std::string& name, const Variant* variant, std::string& error) {
    error.clear();
    Backend next;
    if (name == "Classical") next = Backend::Classical;
    else if (name == "Auto") next = Backend::Auto;
    else if (name == "StocklessNNUE") {
        if (!compatible_variant(variant)) {
            error = "StocklessNNUE requires a successfully loaded compatible EvalFile";
            return false;
        }
        next = Backend::Neural;
    } else {
        error = "unsupported evaluation backend";
        return false;
    }
    if (selected_backend != next) {
        selected_backend = next;
        advance_generation();
    }
    return true;
}

void variant_changed(const Variant* variant) {
    if (selected_backend == Backend::Neural && !compatible_variant(variant))
        selected_backend = Backend::Classical;
    advance_generation();
}

const char* backend_name() {
    return selected_backend == Backend::Classical ? "Classical"
         : selected_backend == Backend::Auto ? "Auto" : "StocklessNNUE";
}

std::string loaded_file() {
    return model ? model->file : "<empty>";
}

std::string status_json(const Variant* variant) {
    std::ostringstream stream;
    stream << "{\"backend\":" << quote(backend_name())
           << ",\"loaded\":" << (model ? "true" : "false")
           << ",\"active\":" << (enabled() && compatible_variant(variant) ? "true" : "false")
           << ",\"compatible\":" << (compatible_variant(variant) ? "true" : "false")
           << ",\"generation\":" << network_generation
           << ",\"variant\":" << quote(variant_name(variant))
           << ",\"model_variant\":" << quote(model ? model->descriptor.name : "")
           << ",\"model_format_version\":" << (model ? (model->descriptor.v2 ? 2 : 1) : 0)
           << ",\"rules_sha256\":" << quote(STOCKLESS_NNUE_RULES_SHA256)
           << ",\"file\":" << quote(model ? model->file : "")
#if defined(USE_AVX2)
           << ",\"accumulator_implementation\":\"avx2\""
#else
           << ",\"accumulator_implementation\":\"scalar\""
#endif
           << '}';
    return stream.str();
}

bool applicable(const Position& pos) {
    return enabled() && compatible_position(pos) && ensure_accumulator(pos);
}

Value evaluate(const Position& pos) {
    if (!applicable(pos))
        return VALUE_ZERO;
    return engine_value(pos, white_cp(numerator(pos, pos.stockless_nnue_accumulator())));
}

Probe probe(const Position& pos, bool compare_refresh) {
    Probe result;
    result.generation = network_generation;
    result.compatible = compatible_position(pos);
    const auto previous_generation = pos.stockless_nnue_accumulator().generation;
    result.used = applicable(pos);
    if (!result.used) {
        result.reason = !model ? "no model loaded" : !result.compatible ? "variant or position mode incompatible"
                      : !enabled() ? "classical backend selected" : "unsupported or oversized feature state";
        return result;
    }
    const auto& accumulator = pos.stockless_nnue_accumulator();
    result.refreshed = previous_generation != network_generation;
    result.accumulator = accumulator.values;
    result.numerator = numerator(pos, accumulator);
    result.white_cp = white_cp(result.numerator);
    result.stm_cp = pos.side_to_move() == WHITE ? result.white_cp : -result.white_cp;
    result.value = engine_value(pos, result.white_cp);
    if (compare_refresh) {
        Accumulator full{};
        const bool valid = refresh(pos, full);
        result.refreshed_accumulator = full.values;
        result.accumulator_matches = valid && result.accumulator == full.values
                                  && accumulator.feature_mass == full.feature_mass;
    }
    return result;
}

std::string evaluation_json(const Position& pos) {
    const auto result = probe(pos);
    std::ostringstream stream;
    stream << "{\"backend\":" << quote(backend_name())
           << ",\"model_variant\":" << quote(model ? model->descriptor.name : "")
           << ",\"active\":" << (enabled() && result.compatible ? "true" : "false")
           << ",\"used\":" << (result.used ? "true" : "false")
           << ",\"compatible\":" << (result.compatible ? "true" : "false")
           << ",\"reason\":" << quote(result.reason)
           << ",\"generation\":" << result.generation
           << ",\"refreshed\":" << (result.refreshed ? "true" : "false")
           << ",\"accumulator_matches\":" << (result.accumulator_matches ? "true" : "false")
           << ",\"numerator\":" << result.numerator
           << ",\"white_cp\":" << result.white_cp << ",\"stm_cp\":" << result.stm_cp
           << ",\"engine_value\":" << int(result.value) << ",\"accumulator\":";
    json_array(stream, result.accumulator);
    stream << ",\"refreshed_accumulator\":";
    json_array(stream, result.refreshed_accumulator);
    stream << '}';
    return stream.str();
}

void board_delta(const Position& pos, Piece pc, Square square, int amount) {
    if (!model) {
        pos.stockless_nnue_accumulator().generation = 0;
        return;
    }
    const int file = int(file_of(square)), rank = int(rank_of(square));
    if (file < 0 || file >= model->descriptor.width || rank < 0 || rank >= model->descriptor.height) {
        pos.stockless_nnue_accumulator().generation = 0;
        return;
    }
    delta(pos, pc, rank * model->descriptor.width + file, amount);
}

void pocket_delta(const Position& pos, Piece pc, int amount) {
    if (!model) {
        pos.stockless_nnue_accumulator().generation = 0;
        return;
    }
    delta(pos, pc, model->descriptor.area, amount);
}

} // namespace Stockfish::Stockless::Neural
