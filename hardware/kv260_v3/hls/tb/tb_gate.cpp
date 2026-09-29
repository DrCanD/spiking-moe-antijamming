#include "moe_top.h"
#include "build_id.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

static int window_data[WIN_WORDS];
static unsigned int results_data[RES_WORDS];
template <class T> std::vector<T> read_file(const std::string &p) {
    std::ifstream f(p.c_str(),std::ios::binary);
    if (!f) { std::cerr << "cannot open " << p << std::endl; std::exit(2); }
    f.seekg(0,std::ios::end);size_t n=(size_t)f.tellg();f.seekg(0);
    if (n%sizeof(T)) std::exit(3);
    std::vector<T> v(n/sizeof(T));f.read((char*)v.data(),n);return v;
}
template <class T> void save_file(const std::string &p,const T *v,size_t n) {
    std::ofstream f(p.c_str(),std::ios::binary);f.write((const char*)v,n*sizeof(T));
}
int main(int argc,char **argv) {
    if (argc<3) { std::cerr<<"tb_gate vector_dir name [name ...] [--generate]"<<std::endl;return 2; }
    std::string dir=argv[1];bool generate=false;
    for(int j=2;j<argc;j++) if(std::string(argv[j])=="--generate")generate=true;
    for(int j=2;j<argc;j++) {
        std::string name=argv[j];if(name=="--generate")continue;
        auto x=read_file<int16_t>(dir+"/"+name+".bin");int n=(int)x.size();
        if(n<2048||n>N_MAX||n%20){std::cerr<<"invalid vector size";return 2;}
        for(int c=0;c<(n+WIN_WORDS-1)/WIN_WORDS;c++) {
            for(int k=0;k<WIN_WORDS;k++)window_data[k]=(c*WIN_WORDS+k<n)?x[c*WIN_WORDS+k]:0;
            moe_top(CMD_LOAD,c,0,0,0,n,window_data,results_data);
        }
        // modes 0..2 produce the expected files; every power-saving variant must reproduce its base mode exactly:
        // dense+ROM isolation == dense, gate+operand isolation(+ROM isolation)(+whole-bank skip) == gate
        const int NM=7;
        const int modes[NM]={EN_FE,EN_FE|EN_GATE,EN_CONV,EN_FE|EN_GATE|EN_ISO,EN_FE|EN_ROMISO,EN_FE|EN_GATE|EN_ISO|EN_ROMISO,
                             EN_FE|EN_GATE|EN_ISO|EN_ROMISO|EN_BANK};
        const char *labels[NM]={"dense","gate","fft","gate","dense","gate","gate"};
        const char *names[NM]={"dense","gate","fft","gate_iso","dense_rom","gate_iso_rom","gate_iso_rom_bank"};
        for(int m=0;m<NM;m++) {
            moe_top(CMD_VERIFY,0,m==2?CAP_NONE:CAP_SPIKES,modes[m],name=="rtl_smoke"?3:0,n,window_data,results_data);
            if(results_data[R_MAGIC]!=RES_MAGIC||results_data[R_BUILD_ID]!=GATE_BUILD_ID)return 3;
            std::string stem=dir+"/"+name+"_"+labels[m];
            if(generate && m<3) save_file(stem+"_res.bin",results_data,RES_WORDS);
            else {                                   // (in --generate mode gate_iso is still checked against the gate file just written)
                auto expected=read_file<uint32_t>(stem+"_res.bin");
                for(int k=0;k<RES_WORDS;k++) {
                    bool compare=m!=2 ? (k<385 || (k>=R_GATE_UPDATES&&k<=R_GATE_LONG)) : (k>=R_CONV_SX&&k<=R_CONV_P+512);
                    if(compare && results_data[k]!=expected[k]) {
                        std::cerr<<name<<" "<<names[m]<<" word="<<k<<" got="<<results_data[k]<<" expected="<<expected[k]<<std::endl;return 4;
                    }
                }
            }
            if(m!=2) {
                auto exp=read_file<int16_t>(dir+"/"+name+"_spikes.bin");
                for(int c=0;c<(n+WIN_WORDS-1)/WIN_WORDS;c++) {
                    moe_top(CMD_READ,c,0,modes[m],name=="rtl_smoke"?3:0,n,window_data,results_data);
                    for(int k=0;k<WIN_WORDS&&c*WIN_WORDS+k<n;k++)
                        if((int16_t)window_data[k]!=exp[c*WIN_WORDS+k])return 5;
                }
            }
            if(name=="rtl_smoke") {
                std::vector<unsigned int> saved(results_data,results_data+RES_WORDS);
                moe_top(CMD_RUN,2,0,modes[m],3,n,window_data,results_data);
                if(results_data[R_FRAMES_DONE]!=2)return 6;
                for(int k=0;k<RES_WORDS;k++) {
                    bool compare=m!=2 ? (k<385 || (k>=R_GATE_UPDATES&&k<=R_GATE_LONG)) : (k>=R_CONV_SX&&k<=R_CONV_P+512);
                    if(compare && results_data[k]!=saved[k])return 7;
                }
            }
            std::cout<<"PASS "<<name<<" "<<names[m];
            if(modes[m]&EN_BANK) std::cout<<" bank_skips="<<results_data[R_GATE_BANKSKIP];
            std::cout<<std::endl;
        }
    }
    return 0;
}
