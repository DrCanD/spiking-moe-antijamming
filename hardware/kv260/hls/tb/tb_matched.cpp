// Native matched-block captures and combination checks against the frozen HLS IP.
#include "moe_top.h"
#include "moe_params.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
static int window_data[WIN_WORDS];
static unsigned int results_data[RES_WORDS];
template<class T> std::vector<T> read_file(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f)throw std::runtime_error("cannot read "+path);
    auto size=f.tellg();f.seekg(0);std::vector<T> v(static_cast<size_t>(size)/sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()),size);return v;
}
template<class T> void save_file(const std::string& path,const std::vector<T>& v) {
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
}
static std::vector<int16_t> capture(int selector,int mode,int n,int size) {
    moe_top(CMD_VERIFY,0,selector,mode,0,n,window_data,results_data);
    std::vector<int16_t> v(size);
    for(int chunk=0;chunk<(size+WIN_WORDS-1)/WIN_WORDS;chunk++) {
        moe_top(CMD_READ,chunk,0,mode,0,n,window_data,results_data);
        for(int k=0;k<WIN_WORDS&&chunk*WIN_WORDS+k<size;k++)v[chunk*WIN_WORDS+k]=static_cast<int16_t>(window_data[k]);
    }
    return v;
}
static void check_words(const std::vector<uint32_t>& reference,int a,int z) {
    for(int k=a;k<z;k++)if(results_data[k]!=reference[k])throw std::runtime_error("combination mismatch at word "+std::to_string(k));
}
int main(int argc,char** argv) {
    if(argc<3)return 2;
    std::string dir=argv[1];bool generate=false;
    for(int i=2;i<argc;i++)if(std::string(argv[i])=="--generate")generate=true;
    const int FE=EN_FE|EN_GATE|EN_ISO|EN_ROMISO|EN_BANK;
    for(int i=2;i<argc;i++) {
        std::string name=argv[i];if(name=="--generate")continue;
        auto x=read_file<int16_t>(dir+"/"+name+".bin");int n=x.size(),groups=n/BLANK_WINDOW;
        for(int c=0;c<(n+WIN_WORDS-1)/WIN_WORDS;c++) {
            for(int k=0;k<WIN_WORDS;k++)window_data[k]=(c*WIN_WORDS+k<n)?x[c*WIN_WORDS+k]:0;
            moe_top(CMD_LOAD,c,0,0,0,n,window_data,results_data);
        }
        const int modes[]={EN_ALE_NB,EN_ALE_SW,EN_BLANK};
        const char* labels[]={"alenb","alesw","blank"};
        std::vector<uint32_t> nb;
        for(int m=0;m<3;m++) {
            int selector=m==0?CAP_E_NB:m==1?CAP_E_SW:CAP_MASK;
            auto cap=capture(selector,modes[m],n,m==2?groups:n);
            if(m==2)for(int sel:{CAP_SS_LO,CAP_SS_MID,CAP_SS_HI}) {
                auto part=capture(sel,modes[m],n,groups);cap.insert(cap.end(),part.begin(),part.end());
            }
            // Use the mask selector for the common result words.
            moe_top(CMD_VERIFY,0,selector,modes[m],0,n,window_data,results_data);
            std::vector<uint32_t> res(results_data,results_data+RES_WORDS);
            std::string stem=dir+"/"+name+"_"+labels[m];
            if(generate){save_file(stem+"_res.bin",res);save_file(stem+"_cap.bin",cap);}
            else if(read_file<uint32_t>(stem+"_res.bin")!=res||read_file<int16_t>(stem+"_cap.bin")!=cap)throw std::runtime_error("capture mismatch "+stem);
            if(m==0)nb=res;
        }
        for(int mode:{FE,static_cast<int>(EN_CONV)}) {
            moe_top(CMD_VERIFY,0,CAP_NONE,mode,0,n,window_data,results_data);
            std::vector<uint32_t> base(results_data,results_data+RES_WORDS);
            moe_top(CMD_VERIFY,0,CAP_NONE,mode|EN_ALE_NB,0,n,window_data,results_data);
            if(mode==FE){check_words(base,0,385);check_words(base,979,982);}else check_words(base,394,958);
            check_words(nb,385,389);check_words(nb,392,393);
        }
        std::cout<<"PASS "<<name<<" ALE_NB/ALE_SW/blanker/combinations"<<std::endl;
    }
    return 0;
}
