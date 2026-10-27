// Stockless Variants contributors, 2026. SPDX-License-Identifier: AGPL-3.0-only
// Experimental independent king-bucket, dual-perspective SVNNUE-v3 Lite.
// Network model format is NOT compatible with Stockfish .nnue or SVNNUE-v1/v2.
#include "v3.h"
#include "nnue.h"
#include "nnue_rules.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "../position.h"
#include "../variant.h"

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

namespace Stockfish::Stockless::Neural::V3 {
namespace {

constexpr std::size_t MaximumFileBytes = 32 * 1024 * 1024;
constexpr std::uint8_t Magic[8] = {'S','V','N','N','U','E','3',0};
constexpr std::uint8_t MagicQ[8] = {'S','V','N','N','U','E','3','Q'};
constexpr std::int32_t HighFtFactor = 4, HighTailFactor = 2;
constexpr std::int32_t MaxFeatureMass = 512;
constexpr std::uint32_t FormatVersion = 3;
constexpr std::uint32_t FeatureVersion = 3;
static_assert(std::is_trivially_copyable<Accumulator>::value, "Per-Position v3 cache must be POD");

struct Descriptor {
    std::string name;
    int width=0, height=0, area=0;
    std::vector<int> types;
    std::array<int, PIECE_TYPE_NB> type_index{};
    std::uint32_t dimensions=0;
    bool rotate_black=false;
};

struct Model {
    const Variant* variant=nullptr;
    Descriptor descriptor;
    std::int32_t ft_factor=1, tail_factor=1;
    std::vector<std::int16_t> transformer; // [feature][128], shared by two perspectives
    std::array<std::int32_t, Hidden> ft_bias{};
    std::array<std::int16_t, Hidden*2*L1> first{}; // [2*128][32]
    std::array<std::int32_t, L1> first_bias{};
    std::array<std::int16_t, L1*L2> second{}; // [32][16]
    std::array<std::int32_t, L2> second_bias{};
    std::array<std::int16_t, L2> output{};
    std::int32_t output_bias=0;
    std::string name;
};

std::unique_ptr<Model> loaded;
bool selected=false;
std::uint64_t generation=1;

void bump() { if (++generation==0) ++generation; }

std::string quote(const std::string& value) {
    static constexpr char digits[]="0123456789abcdef";
    std::string out="\"";
    for (unsigned char ch : value) {
        if (ch=='"' || ch=='\\') { out+='\\';out+=char(ch); }
        else if (ch<32) {out+="\\u00";out+=digits[ch>>4];out+=digits[ch&15];}
        else out+=char(ch);
    }
    return out+'"';
}

std::uint32_t load_u32(const std::uint8_t* data) {
    return std::uint32_t(data[0]) | (std::uint32_t(data[1])<<8)
         | (std::uint32_t(data[2])<<16) | (std::uint32_t(data[3])<<24);
}

std::int32_t load_i32(const std::uint8_t* data) {
    std::uint32_t x=load_u32(data);
    return x<=std::uint32_t(std::numeric_limits<std::int32_t>::max())
         ? std::int32_t(x) : std::int32_t(std::int64_t(x)-(INT64_C(1)<<32));
}

std::int16_t load_i16(const std::uint8_t* data) {
    const std::uint32_t x=std::uint32_t(data[0]) | (std::uint32_t(data[1])<<8);
    return std::int16_t(x<=32767 ? std::int32_t(x) : std::int32_t(x)-65536);
}

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


bool rules_hash(std::array<std::uint8_t,32>& result) {
    const std::string source=STOCKLESS_NNUE_RULES_SHA256;
    if(source.size()!=64) return false;
    const auto digit=[](char ch)->int {
        if(ch>='0' && ch<='9')return ch-'0';
        if(ch>='a' && ch<='f')return ch-'a'+10;
        if(ch>='A' && ch<='F')return ch-'A'+10;
        return -1;
    };
    for (unsigned i=0;i<32;++i) {
        const int hi=digit(source[i*2]),lo=digit(source[i*2+1]);
        if(hi<0 || lo<0)return false;
        result[i]=std::uint8_t(16*hi+lo);
    }
    return true;
}

Descriptor describe(const Variant* variant) {
    Descriptor d;
    d.type_index.fill(-1);
    d.name=Stockless::Neural::variant_name(variant);
    if(d.name!="chess" && d.name!="shogi" && d.name!="janggi"
       && d.name!="xiangqi" && d.name!="makruk") {
        d.name.clear();
        return d;
    }
    d.width=int(variant->maxFile)+1;
    d.height=int(variant->maxRank)+1;
    d.area=d.width*d.height;
    for(int type=int(PAWN);type<=int(KING);++type)
        if(variant->pieceTypes & PieceType(type)) {
            d.type_index[type]=int(d.types.size());
            d.types.push_back(type);
        }
    if(d.types.empty() || d.width<1 || d.width>12 || d.height<1 || d.height>10)
        {d.name.clear();return d;}
    const auto layer_size=2u*std::uint32_t(d.types.size())*std::uint32_t(d.area+1);
    d.dimensions=KingBuckets*layer_size;
    d.rotate_black=(d.name=="shogi");
    return d;
}

bool consistent(const Position& pos) {
    return loaded && loaded->variant==pos.variant() && !pos.is_chess960();
}

int orient_file(const Descriptor& d, Color perspective,int file) {
    return perspective==BLACK && d.rotate_black ? d.width-1-file : file;
}
int orient_rank(const Descriptor& d, Color perspective,int rank) {
    return perspective==BLACK ? d.height-1-rank : rank;
}

int royal_bucket(const Position& pos, const Descriptor& d, Color perspective) {
    const PieceType royal=pos.nnue_king() ? pos.nnue_king() : KING;
    if(pos.count(perspective,royal)!=1)
        return -1; // Never invent a non-existent royal anchor.
    const Square king=pos.square(perspective,royal);
    if(king==SQ_NONE) return -1;
    const int file=orient_file(d,perspective,int(file_of(king)));
    const int rank=orient_rank(d,perspective,int(rank_of(king)));
    if(file<0 || file>=d.width || rank<0 || rank>=d.height)
        return -1;
    return 4*(4*rank/d.height)+(4*file/d.width);
}

int feature_index(const Descriptor& d,Color perspective,int bucket,Piece pc,int slot) {
    if(pc==NO_PIECE || int(pc)>=int(PIECE_NB) || bucket<0 || bucket>=int(KingBuckets))
        return -1;
    const int kind=int(type_of(pc));
    if(kind<0 || kind>=PIECE_TYPE_NB || d.type_index[kind]<0)
        return -1;
    const int relative=(color_of(pc)==perspective)?0:1;
    if(slot<0 || slot>d.area)return -1;
    const int index=(((bucket*2+relative)*int(d.types.size())+d.type_index[kind])
                     *(d.area+1)+slot);
    return index>=0 && std::uint32_t(index)<d.dimensions?index:-1;
}

bool extract(const Position& pos,const Descriptor& d,Color perspective,
             std::vector<std::pair<std::uint32_t,std::int32_t>>& result,
             int& mass,int& bucket) {
    mass=0;
    bucket=royal_bucket(pos,d,perspective);
    if(bucket<0) return false;
    for(int rank=0;rank<d.height;++rank)
        for(int file=0;file<d.width;++file) {
            const Piece pc=pos.piece_on(make_square(File(file),Rank(rank)));
            if(pc==NO_PIECE)continue;
            const int slot=orient_rank(d,perspective,rank)*d.width
                            +orient_file(d,perspective,file);
            const int index=feature_index(d,perspective,bucket,pc,slot);
            if(index<0 || mass>=MaxFeatureMass) return false;
            result.emplace_back(std::uint32_t(index),1);
            ++mass;
        }
    for(Color color : {WHITE,BLACK})
        for(int kind : d.types) {
            const int count=pos.count_in_hand(color,PieceType(kind));
            if(count<0 || count>MaxFeatureMass-mass)return false;
            if(count) {
                const Piece pc=make_piece(color,PieceType(kind));
                const int index=feature_index(d,perspective,bucket,pc,d.area);
                if(index<0) return false;
                result.emplace_back(std::uint32_t(index),count);
                mass+=count;
            }
        }
    std::sort(result.begin(),result.end());
    return true;
}

void add(std::array<std::int32_t,Hidden>& acc,const std::int16_t* weight,int count) {
#if defined(USE_AVX2)
    const __m256i mult=_mm256_set1_epi32(count);
    for(unsigned k=0;k<Hidden;k+=8) {
        const __m128i packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(weight+k));
        const __m256i expanded=_mm256_cvtepi16_epi32(packed);
        const __m256i old=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data()+k));
        const __m256i next=_mm256_add_epi32(old,_mm256_mullo_epi32(expanded,mult));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data()+k),next);
    }
#else
    for(unsigned k=0;k<Hidden;++k)
        acc[k]+=std::int32_t(weight[k])*count;
#endif
}

bool refresh(const Position& pos, Accumulator& state) {
    if(!consistent(pos))return false;
    for(Color color : {WHITE,BLACK}) {
        std::vector<std::pair<std::uint32_t,std::int32_t>> data;
        int mass=0,bucket=-1;
        if(!extract(pos,loaded->descriptor,color,data,mass,bucket))
            { state.generation=0;return false; }
        auto& dst=state.values[int(color)];
        dst=loaded->ft_bias;
        for(const auto& item : data)
            add(dst,loaded->transformer.data()
                    +std::size_t(item.first)*Hidden,item.second);
        state.bucket[int(color)]=std::uint32_t(bucket);
        state.mass[int(color)]=mass;
    }
    state.generation=generation;
    return true;
}

bool ensure(const Position& pos) {
    auto& state=pos.stockless_v3_accumulator();
    // All own-royal board changes invalidate generation in board_delta().
    // Hence a valid generation proves both cached anchors are still current.
    // Avoid two royal-square lookups at every searched leaf.
    return state.generation==generation || refresh(pos,state);
}

std::int32_t clamp_act(std::int64_t value) {
    return std::int32_t(std::clamp<std::int64_t>(value,0,QA));
}
std::int32_t clamp_ft(std::int64_t value) {
    // The quantized tail applies its own factor. This preserves the full
    // high-resolution transformer activation until after layer one.
    return std::int32_t(std::clamp<std::int64_t>(
        value,0,std::int64_t(QA)*loaded->ft_factor));
}
std::int64_t round_div(std::int64_t value,std::int64_t divisor) {
    return value>=0 ? (value+divisor/2)/divisor
                    : -((-value+divisor/2)/divisor);
}

std::int32_t stm_centipawns(const Position& pos,const Accumulator& state) {
    std::array<std::int32_t,2*Hidden> act{};
    const int first=int(pos.side_to_move());
    for(unsigned color=0;color<2;++color)
        for(unsigned k=0;k<Hidden;++k)
            act[color*Hidden+k]=clamp_ft(state.values[color==0?first:1-first][k]);

    // The on-disk matrices are row-major. Accumulate each contiguous row
    // into all output neurons, and skip zero clipped activations entirely.
    std::array<std::int64_t,L1> first_acc{};
    for(unsigned j=0;j<L1;++j)
        first_acc[j]=std::int64_t(loaded->first_bias[j])*loaded->ft_factor;
    for(unsigned i=0;i<act.size();++i) {
        if(!act[i])continue;
        const auto* row=loaded->first.data()+i*L1;
        for(unsigned j=0;j<L1;++j)
            first_acc[j]+=std::int64_t(act[i])*row[j];
    }
    std::array<std::int32_t,L1> layer1{};
    for(unsigned j=0;j<L1;++j)
        layer1[j]=clamp_act(round_div(
            first_acc[j],std::int64_t(QB)*loaded->ft_factor*loaded->tail_factor));

    std::array<std::int64_t,L2> second{};
    for(unsigned j=0;j<L2;++j)second[j]=loaded->second_bias[j];
    for(unsigned i=0;i<L1;++i) {
        if(!layer1[i])continue;
        const auto* row=loaded->second.data()+i*L2;
        for(unsigned j=0;j<L2;++j)
            second[j]+=std::int64_t(layer1[i])*row[j];
    }
    std::array<std::int32_t,L2> layer2{};
    for(unsigned j=0;j<L2;++j)
        layer2[j]=clamp_act(round_div(second[j],std::int64_t(QB)*loaded->tail_factor));

    std::int64_t total=loaded->output_bias;
    for(unsigned k=0;k<L2;++k)
        total+=std::int64_t(layer2[k])*loaded->output[k];
    return std::int32_t(std::clamp(round_div(total,std::int64_t(QA)*QB*loaded->tail_factor),
                         std::int64_t(-10000),std::int64_t(10000)));
}

std::string quote_features(const std::vector<std::pair<std::uint32_t,std::int32_t>>& features) {
    std::ostringstream stream;
    stream<<'[';
    bool first=true;
    for(const auto& feature : features) {
        if(!first)stream<<',';
        first=false;
        stream<<'['<<feature.first<<','<<feature.second<<']';
    }
    stream<<']';
    return stream.str();
}

std::size_t expected_payload(const Descriptor& d) {
    return std::size_t(d.dimensions)*Hidden*2 + Hidden*4
        + (2*Hidden*L1)*2 + L1*4 + L1*L2*2 + L2*4 + L2*2 + 4;
}

void delta(const Position& pos, Piece pc,Square square,int amount,bool pocket) {
    auto& state=pos.stockless_v3_accumulator();
    if(state.generation==0)return;
    if(!selected || !consistent(pos) || state.generation!=generation)
        {state.generation=0;return;}
    // A royal move changes that perspective's entire feature index group.
    const PieceType royal=pos.nnue_king()?pos.nnue_king():KING;
    if(type_of(pc)==royal) {state.generation=0;return;}
    const auto& d=loaded->descriptor;
    for(Color perspective : {WHITE,BLACK}) {
        const int slot=pocket?d.area:
                orient_rank(d,perspective,int(rank_of(square)))*d.width
                +orient_file(d,perspective,int(file_of(square)));
        const int index=feature_index(d,perspective,int(state.bucket[int(perspective)]),pc,slot);
        const int mass=state.mass[int(perspective)]+amount;
        if(index<0 || mass<0 || mass>MaxFeatureMass)
            {state.generation=0;return;}
        add(state.values[int(perspective)],loaded->transformer.data()
            +std::size_t(index)*Hidden,amount);
        state.mass[int(perspective)]=mass;
    }
}

} // anonymous namespace

bool supported(const Variant* variant) {return !describe(variant).name.empty();}
bool compatible(const Variant* variant) {
    return loaded && loaded->variant==variant && supported(variant);
}
bool available() {return bool(loaded);}
void activate(bool enabled) {if(selected!=enabled){selected=enabled;bump();}}
void invalidate() {bump();}
std::string file() {return loaded?loaded->name:std::string();}

std::string descriptor_json(const Variant* variant) {
    const auto d=describe(variant);
    std::ostringstream s;
    s<<"{\"schema\":3,\"format_version\":3,\"feature_version\":3,"
     <<"\"variant\":"<<quote(d.name)<<",\"supported\":"<<(d.name.empty()?"false":"true")
     <<",\"input_dim\":"<<d.dimensions<<",\"hidden_dim\":"<<Hidden
     <<",\"l1_dim\":"<<L1<<",\"l2_dim\":"<<L2
     <<",\"qa\":"<<QA<<",\"qb\":"<<QB
     <<",\"king_buckets\":"<<KingBuckets
     <<",\"width\":"<<d.width<<",\"height\":"<<d.height
     <<",\"rules_sha256\":"<<quote(STOCKLESS_NNUE_RULES_SHA256)
     <<",\"black_rotation\":"<<(d.rotate_black?"true":"false")
     <<",\"type_ids\":[";
    for(unsigned i=0;i<d.types.size();++i){if(i)s<<',';s<<d.types[i];}
    s<<"]}";
    return s.str();
}

std::vector<std::pair<std::uint32_t,std::int32_t>>
features(const Position& pos,Color perspective) {
    std::vector<std::pair<std::uint32_t,std::int32_t>> result;
    const auto d=describe(pos.variant());
    int mass=0,bucket=0;
    if(!extract(pos,d,perspective,result,mass,bucket))
        result.clear();
    return result;
}

std::string native_features_json(const Position& pos) {
    const auto d=describe(pos.variant());
    const auto white=features(pos,WHITE),black=features(pos,BLACK);
    if(d.name.empty()||white.empty()||black.empty())
        return "{\"schema\":3,\"error\":\"unsupported or missing royal anchor\"}";
    std::ostringstream s;
    s<<"{\"schema\":3,\"descriptor\":"<<descriptor_json(pos.variant())
     <<",\"side_to_move\":"<<int(pos.side_to_move())
     <<",\"white_bucket\":"<<royal_bucket(pos,d,WHITE)
     <<",\"black_bucket\":"<<royal_bucket(pos,d,BLACK)
     <<",\"features_white\":"<<quote_features(white)
     <<",\"features_black\":"<<quote_features(black)<<"}";
    return s.str();
}

bool load(const std::string& path,const Variant* variant,std::string& error) {
    error.clear();
    const auto d=describe(variant);
    if(d.name.empty()){error="unsupported v3 variant";return false;}
    std::ifstream stream(path,std::ios::binary|std::ios::ate);
    if(!stream){error="cannot open SVNNUE3 file";return false;}
    const auto length=stream.tellg();
    if(length<std::streamoff(HeaderSize) || length>std::streamoff(MaximumFileBytes))
        {error="invalid SVNNUE3 file size";return false;}
    std::vector<std::uint8_t> data(std::size_t(length),std::uint8_t(0));
    stream.seekg(0);
    if(!stream.read(reinterpret_cast<char*>(data.data()),std::streamsize(data.size())))
        {error="truncated SVNNUE3 file";return false;}
    const bool highPrecision=std::equal(std::begin(MagicQ),std::end(MagicQ),data.begin());
    const bool legacyPrecision=std::equal(std::begin(Magic),std::end(Magic),data.begin());
    if(!highPrecision && !legacyPrecision)
        {error="not a versioned SVNNUE3 or SVNNUE3Q network";return false;}
    const std::uint32_t wanted[]={
        FormatVersion,FeatureVersion,Hidden,d.dimensions,
        std::uint32_t(d.width),std::uint32_t(d.height),std::uint32_t(d.types.size()),
        QA,QB,std::uint32_t(expected_payload(d))
    };
    for(unsigned i=0;i<10;++i)
        if(load_u32(data.data()+8+4*i)!=wanted[i])
            {error="SVNNUE3 feature/header architecture mismatch";return false;}
    std::array<std::uint8_t,32> name{};
    if(d.name.size()>31){error="name overflow";return false;}
    std::copy(d.name.begin(),d.name.end(),name.begin());
    if(!std::equal(name.begin(),name.end(),data.begin()+48))
        {error="network targets another game";return false;}
    std::array<std::uint8_t,32> hash{};
    if(!rules_hash(hash)||!std::equal(hash.begin(),hash.end(),data.begin()+80))
        {error="native v3 source/rules fingerprint mismatch";return false;}
    if(std::any_of(data.begin()+144,data.begin()+HeaderSize,
                   [](std::uint8_t x){return x!=0;}))
        {error="nonzero reserved bytes";return false;}
    if(expected_payload(d)+HeaderSize!=data.size())
        {error="network payload truncated or trailing bytes";return false;}
    const auto digest=sha256(data.data()+HeaderSize,expected_payload(d));
    if(!std::equal(digest.begin(),digest.end(),data.begin()+112))
        {error="SVNNUE3 payload sha256 mismatch";return false;}
    auto candidate=std::make_unique<Model>();
    candidate->variant=variant;
    candidate->descriptor=d;
    candidate->ft_factor=highPrecision ? HighFtFactor : 1;
    candidate->tail_factor=highPrecision ? HighTailFactor : 1;
    candidate->name=path;
    candidate->transformer.resize(std::size_t(d.dimensions)*Hidden);
    std::size_t offset=HeaderSize;
    const auto read_short=[&](){const auto x=load_i16(data.data()+offset);offset+=2;return x;};
    const auto read_int=[&](){const auto x=load_i32(data.data()+offset);offset+=4;return x;};
    for(auto& w:candidate->transformer) w=read_short();
    for(auto& b:candidate->ft_bias) {
        b=read_int();
        if(b<-(1<<20)||b>(1<<20))
            {error="unsafe transformer bias";return false;}
    }
    for(auto& w:candidate->first)w=read_short();
    for(auto& b:candidate->first_bias)b=read_int();
    for(auto& w:candidate->second)w=read_short();
    for(auto& b:candidate->second_bias)b=read_int();
    for(auto& w:candidate->output)w=read_short();
    candidate->output_bias=read_int();
    if(offset!=data.size()){error="unexpected SVNNUE3 payload layout";return false;}
    loaded=std::move(candidate);
    bump();
    return true;
}

bool applicable(const Position& pos) {
    return selected && consistent(pos) && ensure(pos);
}

Value evaluate_cached(const Position& pos) {
    // Caller has already checked applicable() in the search evaluation hook.
    const int score=stm_centipawns(pos,pos.stockless_v3_accumulator());
    return Value(score*int(PawnValueEg)/100);
}

Value evaluate(const Position& pos) {
    return applicable(pos) ? evaluate_cached(pos) : VALUE_ZERO;
}

std::string status_json(const Variant* variant) {
    std::ostringstream out;
    out<<"{\"format\":3,\"backend\":\"StocklessNNUEv3\","
       <<"\"active\":"<<(selected&&compatible(variant)?"true":"false")
       <<",\"loaded\":"<<(loaded?"true":"false")
       <<",\"file\":"<<quote(file())
       <<",\"model_variant\":"<<quote(loaded?loaded->descriptor.name:"")
       <<",\"input_dim\":"<<(loaded?loaded->descriptor.dimensions:0)
       <<",\"hidden_dim\":"<<Hidden
       <<",\"precision_mode\":"<<quote(
            loaded && loaded->ft_factor!=1 ? "high" : "legacy")
#if defined(USE_AVX2)
       <<",\"accumulator_implementation\":\"avx2\""
#else
       <<",\"accumulator_implementation\":\"scalar\""
#endif
       <<",\"network_generation\":"<<generation
       <<",\"rules_sha256\":"<<quote(STOCKLESS_NNUE_RULES_SHA256)<<"}";
    return out.str();
}

std::string evaluation_json(const Position& pos) {
    const bool used=applicable(pos);
    int score=used?stm_centipawns(pos,pos.stockless_v3_accumulator()):0;
    bool matches=false;
    if(used) {
        Accumulator full{};
        matches=refresh(pos,full);
        const auto& state=pos.stockless_v3_accumulator();
        matches=matches && state.values==full.values
                && state.mass==full.mass && state.bucket==full.bucket;
    }
    std::ostringstream out;
    out<<"{\"format\":3,\"used\":"<<(used?"true":"false")
       <<",\"accumulator_matches\":"<<(matches?"true":"false")
       <<",\"stm_cp\":"<<score
       <<",\"white_cp\":"<<(pos.side_to_move()==WHITE?score:-score)
       <<",\"generation\":"<<generation<<"}";
    return out.str();
}

void board_delta(const Position& pos,Piece piece,Square square,int amount) {
    delta(pos,piece,square,amount,false);
}
void pocket_delta(const Position& pos,Piece piece,int amount) {
    delta(pos,piece,SQ_A1,amount,true);
}

} // namespace Stockfish::Stockless::Neural::V3
