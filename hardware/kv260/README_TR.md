# UAV rezonatör gate — KV260 deney paketi (v3)

27 Eylül 2026 · UAV anti-jamming / TCCN revizyonu

**Amaç:** mevcut dense rezonatör ön ucunun güncellemelerini, durum güvenli bölgede ve giriş spike'ı yokken atlayan sabit noktalı gate'i KV260 üzerinde sınamak. Önce çıktı doğruluğu, sonra aynı bitstream ve eşleşen hızda fiziksel SOM giriş gücü ölçülür.

**Durum:** kaynaklar ve yerel doğrulama hazır. Bu pakette derlenmiş bitstream yoktur. Burada Vitis/Vivado sentezi, RTL eşbenzetimi veya KV260 güç ölçümü yapılmadı. Bunlar aşağıdaki komutların zorunlu aşamalarıdır.

**Sürüm 3 (27 Eylül 2026) — neden ve ne değişti:**

v2 gece ölçümü (4 vektör × 5 tekrar × 120 s): `FE_gate_iso` dense'e göre fixed_none'da 1,64×, random_pulse'ta 1,36× verimliydi; random_switching ve fixed_narrowband'de 1,11–1,15× daha pahalıydı (başa baş: bant güncellemelerinin ~%62'si atlandığında). SAIF güç dökümü ek yükün kaynağını gösterdi: her çevrim okunan sıçrama tablosu ROM'u (dense modda bile) ve gate'in kayıt tutma mantığı; kart ölçümü çarpıcıların glitch dahil dense'in ~%40–45'i olduğunu gösterdi. v3 bu dört kalemi ele alır:

1. **Dar RF bankası (sayısal değişiklik).** Rotasyon katsayıları Q2.30 (32 bit) yerine **Q1.17 (18 bit)**, durum Q20 (24 bit yazmaç). Her gerçek çarpım 24×18 bit, tek DSP48E2'ye sığar (v2: 28×32). Sıçrama tablosu da 18 bit. Seçim `tools/width_sweep.py` ile yapıldı: kalite protokolünün 288 akışında dondurulmuş Q30/Q20 referansa göre yalnız bu genişlik router kararlarını hiç değiştirmedi (16 bit katsayı veya Q16 durum en az bir kararı değiştirdi; `evidence/width_sweep_summary.json`). Resmî kanıt `evidence/fixed_quality_summary.json`: v3 dense ve v3 gate yolları, 288 akış, 0 yeni hata, 0 değişen karar akışı, 0 fark edilmemiş hata, BLER referansla aynı.
2. **Pipeline'lı çarpıcılar.** Çarpımlar tam kayıtlı DSP'ye bağlanır (`BIND_OP ... latency=3`); glitch zincirleri kesilir. Bant dizileri için `DEPENDENCE inter false` bildirimi II=1'i korur; `build_kv260.py` sentezden sonra RF_BANK'ın elde edilen II'sini yazar. Derleme bayrağı `-DMOE_PIPE_MUL=0` eski bağlamaya döner.
3. **ROM adres yalıtımı (`EN_ROMISO` = 128).** Sıçrama tablosunun adresi yalnız bant atlama yaptığında yüklenir; atlanan bantlarda ve dense modda sabit tutulur. Sonuçlar değişmez.
4. **Tüm bankayı atlama (`EN_BANK` = 256).** Örnekte spike yoksa ve 16 bandın hepsi güvenliyse bant döngüsüne hiç girilmez; sayaçlar 16 bant atlamasıyla aynı ilerler. Sonuçlar `FE_gate` ile bit düzeyinde aynı. Yeni sonuç sözcüğü 984 = banka atlanan örnek sayısı. Bu vektörlerde oran: fixed_none %90, random_pulse %75, random_switching %23, fixed_narrowband ≈ 0.

v3 sihirli sözcüğü `GAT3`, protokol `0x20260927`: kart betikleri v2 bitstream'ini ve tersini kabul etmez. Ölçülen modlar: D0_empty, FE_dense, FE_dense_rom, FE_gate_iso, FE_gate_iso_rom, **FE_v3** (hepsi açık), FFT1024. Doğrulama bunlara ek olarak FE_gate'i de denetler (7 donanım modu × 14 vektör = 98 kontrol).

Yerel doğrulama (v3): C++ ↔ Python v3 kâhini 14 vektörde birebir (dense, gate ve banka-atlama sayacı); açık kaynak `ap_int` başlıklarıyla 4 kopya **98/98 PASS**; g++ yerel 4 kopya 98/98; kart betiğinin `verify_vectors` yolu 4 kopyalı C modeliyle 98/98; 11 ölçüm birim testi. Vitis csim/sentez, Vivado ve kart ölçümü bilgisayarınızda yapılacak.

**Sürüm 2 (26 Eylül 2026, öğleden sonra) — değişiklikler:**

1. **Derleme hatası düzeltildi.** Vitis 2025.2'de `ap_utils.h` yalnızca `include\etc\` altında bulunuyor; ilk sürüm csim'de `'ap_utils.h' file not found` ile durdu. `ap_wait()` C simülasyonunda boş işlem olduğundan başlık artık yalnızca sentezde (`etc/ap_utils.h`, eski sürümlerde üst düzey dosya) çağrılıyor; csim/cosim sonuçları değişmez.
2. **Yeni mod `FE_gate_iso` (`blk_en` = 97).** İlk sürümde gate, II=1 bant döngüsünün içinde `continue` ile yazılmıştı. HLS pipeline döngülerinde yan etkisiz aritmetiği her yinelemede hesaplar, yalnızca yazmaları bastırır; çarpıcı girişleri atlanan bantlarda da değişmeye devam eder (bkz. §6). `FE_gate_iso` bu girişleri atlanan bantlarda registerda sabit tutar (operand yalıtımı). Çıktıları `FE_gate` ile bit düzeyinde aynıdır; test tezgâhı ve kart doğrulaması bunu her vektörde denetler. `FE_gate` (teslim edildiği hâliyle) ölçüm planında kalır.
3. Kaynak kimliği yenilendi, `prepare_vectors.py` yeniden çalıştırıldı: giriş vektörleri ve spike dosyaları bit düzeyinde aynı; beklenen sonuç dosyalarında yalnızca build-ID (983) ve kopya-0 sağlama toplamı (958) sözcükleri değişti — ikisi de karşılaştırma kümesinde değil.
4. `requirements-quality.txt` dosyasına eksik `reedsolo==1.7.0` eklendi.
5. Yerel doğrulama: AMD'nin açık kaynak `ap_int` başlıklarıyla (clang, `-DUSE_AP_INT -DN_REPL=4`) 14 vektör × 4 mod = **56/56 PASS**; aynı sonuç g++ yerel tamsayı yapısında; kart betiğinin `verify_vectors` yolu C modeli üzerinden 56/56; 9 ölçüm birim testi geçti. Vitis csim/sentez/cosim ve Vivado hâlâ bilgisayarınızda yapılacak.

## 1. Bilgisayarda derleme

Vivado/Vitis kurulu **x86 bilgisayarda** çalıştırın. Bu adım Colab'da veya KV260'ın ARM işlemcisinde çalıştırılmaz. Varsayılan akış mevcut projenizin Vitis/Vivado 2025.2 unified HLS düzenine göre hazırlanmıştır; ilgili KV260 SOM/carrier board dosyaları kurulu olmalıdır. Python 3.10+ yeterlidir; derleme sürücüsü yalnızca standart kütüphaneyi kullanır.

ZIP'i kısa ve boşluksuz bir dizine çıkarın. AMD araçlarının ortamını açtıktan sonra paketin `kv260_gate` klasörüne geçin:

```bash
python build_kv260.py --jobs 4
```

Linux'ta Python komutunuz `python3` ise onu kullanın. Windows'ta Vitis ve Vivado'nun `settings64.bat` dosyalarını aynı komut penceresinde `call` ile yükleyin; kurulum yolu kendi sisteminizdeki yol olmalıdır. `vitis-run`, `v++`, `vivado` ve `bootgen` bu pencereden bulunabilmelidir.

Sürücü sırasıyla şunları yapar:

1. Kaynak kimliğini ve mevcut sabit noktalı kalite raporunu denetler.
2. Gerçek `ap_int` tipleriyle 14 vektörde C simülasyonu yapar.
3. HLS sentezi ve `rtl_smoke` vektöründe RTL eşbenzetimi yapar. Bu vektör pozitif/negatif spike, sessizlik, 256 örneği aşan uyku ve yeniden uyanmayı içerir; VERIFY ve iki çerçeveli RUN yolları karşılaştırılır.
4. HLS IP'sini paketler; aynı seçilmiş IP ile Vivado implementasyonu ve bitstream üretir.
5. Setup/hold zamanlama sonucu uygun ise gerçek HLS başlığından register haritasını çıkarır, firmware'i ve `board/build_receipt.json` kaydını oluşturur.

Her aşamanın çıktısı `build_logs` altında tutulur. Uzun derleme sırasında bu dosyaların büyümesi normaldir. Süre bilgisayara ve araçların uyguladığı tasarıma bağlıdır. Hata olursa `build_logs` klasörünü gönderin; adımları atlayarak devam etmeyin.

`vitis_hls` bulunan daha eski bir kurulum için alternatif:

```bash
python build_kv260.py --flow classic --jobs 4
```

Araç sürümleriyle uyum henüz yerinde sınanmadığından bu alternatifin başarı garantisi yoktur. Kaynak, clock veya `N_REPL` değerini ilk deneyden önce değiştirmeyin; paket kaynak kimliği ve doğrulama kayıtları bunlara bağlıdır.

## 2. KV260'a aktarma ve yükleme

Derleme tamamlandığında **paketin tamamını**, üretilen `board/firmware`, `board/regmap.json` ve `board/build_receipt.json` dosyalarıyla birlikte KV260'a kopyalayın. Mevcut Kria Ubuntu / `xmutil` düzeni hedeflenmiştir.

Kart üzerinde Python 3, NumPy, `dtc` ve `xmutil` bulunmalıdır. NumPy ve device-tree compiler eksikse mevcut Ubuntu paket yöneticinizle `python3-numpy` ve `device-tree-compiler` kurabilirsiniz. Kartta kalite deneyinin scikit-learn/Numba bağımlılıkları gerekmez.

Kartta paketin `kv260_gate` klasöründen:

```bash
sudo bash board/install_gate.sh
sudo python3 board/run_gate.py --verify-only
```

Yükleyici çalışan FPGA uygulamasını kaldırıp `moe_gate` uygulamasını yükler. Eski `moe` firmware dosyalarını silmez. `--verify-only`, 14 vektörde encoder spike'larını, dense/gated ön uç sayaçlarını ve FFT sonuçlarını beklenen çıktılarla karşılaştırır. Yanlış bitstream, kaynak kimliği, register haritası veya gerçek PL saat hızı kabul edilmez.

## 3. Kısa ölçüm, ardından gece koşusu

```bash
sudo python3 board/run_gate.py --quick
```

Kısa ölçüm kurulum teşhisidir; güven aralığıyla enerji üstünlüğü iddiası üretmez. Her çalıştırmada kart doğrulaması tekrar yapılır.

Çalışırken ekranda ne yapıldığı özetlenir: `[PLAN]` (toplam süre), `[KAL ]` (her kalibrasyon denemesi: pace → gerçekleşen hız), `[KOSU]`/`[BOS ]` (blok içinde ~10 sn'de bir: vektör, tekrar, mod, anlık ve ortalama güç, son boşa göre fark, gerçekleşen hız, sıcaklık, genel ilerleme ve kalan süre), `[POWER]` (her koşunun boşa göre farkı, kopya başına mW ve nJ/örnek) ve her tekrar sonunda `[OZET]` (D0 çıkarılmış kopya başına nJ/örnek ve eşleştirilmiş farklar).

Kısa koşu başarılıysa:

```bash
sudo python3 board/run_gate.py --overnight
```

Varsayılan uzun koşu: 4 giriş vektörü × 5 tekrar × 7 mod (v3); aktif/boş referans blokları 120 saniye. Bekleme ve kalibrasyonlarla **yaklaşık 10,5–11,5 saat** planlayın. SSH bağlantınız kapanacaksa komutu açık kalacak `tmux` oturumunda çalıştırın. Soğutma/fan ayarını ve güç beslemesini koşu boyunca sabit tutun; diğer ağır işleri kapatın.

Sonuç:

```text
board/results/<zaman>/KV260_GATE_RESULTS.zip
```

Bu ZIP'i gönderin. İçinde ham güç örnekleri, sıcaklıklar, gerçekleşen hızlar, mod sırası, kalibrasyon, çıktı doğrulaması, istatistikler ve implementasyon raporları bulunur. Hatalı koşu da `failure.json` içeren ZIP bırakır. Her bloktan sonra ara sonuçlar yazılır; mevcut sürüm yarım koşuyu otomatik sürdürmez.

## 4. Örnekleme hızı koşulu

Hedef **kopya başına 1 MS/s**. Her mod önce en yüksek hızıyla denenir; aynı hedefe getirilemeyen veya ölçüm sırasında hızları %1'den fazla ayrışan modların karşılaştırması durdurulur. Hedef sessizce düşürülmez.

1 MS/s sağlanamazsa hata ZIP'ini gönderin. Yalnızca daha düşük throughput'ta ön karakterizasyon yapmak isterseniz bunu açıkça seçebilirsiniz:

```bash
sudo python3 board/run_gate.py --quick --target 0.5
sudo python3 board/run_gate.py --overnight --target 0.5
```

0.5 MS/s koşusu, 1 MHz giriş dalga şeklinin gerçek zamanlı çalıştığını kanıtlamaz. 1 MS/s ortalama replay hızının sağlanması da sürekli ADC akışında en kötü durum gecikmesini kanıtlamaz: bu paket URAM'daki çerçeveyi tekrar oynatan bir ölçüm düzeneğidir; gerçek ADC/DMA giriş ve akış tamponlaması ayrıca sınanmalıdır.

## 5. Tam olarak neyi karşılaştırıyoruz?

| Mod | `blk_en` | İçerik |
|---|---:|---|
| D0_empty | 0 | Replay, kontrol, sonuç yazımı ve hız beklemesi referansı |
| FE_dense | 1 | Delta encoder + spike istatistikleri + 16 dense rezonatör |
| FE_gate | 33 | Aynı ön uç + güvenli durum kontrolü + gecikmeli güncelleme tablosu (teslim edildiği hâliyle; atlanan bantlarda çarpıcı girişleri değişmeye devam eder) |
| FE_gate_iso | 97 | FE_gate + operand yalıtımı: atlanan bantlarda çarpıcı girişleri sabit tutulur; çıktılar FE_gate ile bit düzeyinde aynı |
| FE_dense_rom | 129 | v3: FE_dense + sıçrama tablosu adresi sabit (dense tabloyu kullanmaz); çıktılar FE_dense ile aynı |
| FE_gate_iso_rom | 225 | v3: FE_gate_iso + ROM adres yalıtımı; çıktılar FE_gate ile aynı |
| FE_v3 | 481 | v3: FE_gate_iso_rom + tüm bankayı atlama; çıktılar FE_gate ile aynı |
| FFT1024 | 2 | Önceki sabit noktalı Hann/Welch FFT-1024 ve klasik istatistikler |

Aynı yerleştirme/yönlendirme içinde dört kopya (`N_REPL=4`) bulunur. Modlar register ile seçilir; donanım her karşılaştırmada yeniden yüklenmez. Tekrarlarda mod sırası sabit tohumla rastgeleleştirilir. Birincil karşılaştırmalar dense − gate_iso (kavramın etkisi) ve gate − gate_iso (operand yalıtımının etkisi) farklarıdır; dense − gate, teslim edilen gerçeklemenin etkisidir. FFT karşılaştırması yardımcı ön uç maliyet karşılaştırmasıdır; burada FFT ile eşit uçtan uca alıcı kalitesi yeniden kanıtlanmış değildir.

**Bu sürüm eski LIF'i veya bütün v5 alıcıyı FPGA'ya taşımaz.** v5 karar mantığı/uzmanlar/RS+CRC'nin tam donanım gücü, RF/ADC, haberleşme yükü ve UAV uçuş gücü bu ön uç ölçümüne dahil değildir. Kaynakta önceki ALE/blanker blokları korunmuştur ancak ana ölçüm modlarında etkin değildir; “tam v5 enerji sonucu” olarak adlandırılmamalıdır.

## 6. Gate'in sabit noktalı davranışı

- Giriş Q6.10, durum 28 bit / 20 kesir biti, rotasyon katsayıları Q2.30.
- Encoder eşikleri önceki HLS ile aynı: ±2048, refractory=2; eşikte `>=` / `<=`.
- Giriş spike'ı yokken `|Re(z)| + |Im(z)| < threshold - 64 LSB` ise ilgili rezonatör güncellemesi atlanır.
- Yeniden ihtiyaç duyulduğunda son materyalize durumdan itibaren geçen örnek sayısıyla Q30 katsayı tablosundan atlama yapılır. 256 örneği aşan boşlukta eski katkı sıfırlanır.
- 16 × 257 × 2 katsayı, kopya başına ham olarak 32,896 byte eder. Dört kopya için paylaşılmadan 128.5 KiB; gerçek BRAM/DSP/LUT kullanımı sentezde görülecek.

**HLS'de atlamanın gerçek anlamı.** `FE_gate` modunda atlanan bantta durum ve sayaç yazmaları bastırılır, fakat II=1 döngüde çarpımlar (yan etkisiz aritmetik) her yinelemede hesaplanır ve çarpıcı girişleri (`zr[b]`, rotasyon katsayısı) bant değiştikçe değişir. HLS'nin bu spekülatif yürütmesini modelleyen C düzeyi göstergede (çarpıcı girişlerindeki bit değişimi, yineleme başına) `FE_gate` dense ile aynı düzeydeydi (ör. fixed_narrowband 73 → 79, fixed_none 91 → 76); `FE_gate_iso` ile atlanan bant oranıyla birlikte düştü (fixed_narrowband 43.5, fixed_pulse 14.3, fixed_none 4.7; atlanan oranlar %47, %83, %94). Bu bir anahtarlama göstergesidir, güç ölçümü değildir; sentezlenen devrenin davranışını gece ölçümü gösterecek.

Bu, HLS'de **hesaplama/durum güncellemesi atlama** uygulamasıdır. Saat ağı kapatılmış veya fiziksel güç alanı kesilmiş değildir. Clock-enable çıkarımı, çarpıcıdaki gerçek anahtarlama azalması ve gate/LUT maliyeti implementasyon ve güç ölçümüyle değerlendirilecektir. İşlem azalması tek başına enerji kazancı sayılmaz.

Tablodan çoklu adım atlamak, her örnekte yapılan tamsayı kesmelerini bire bir tekrar etmez. Bu nedenle fixed-point gate, dense sürümle tüm durum ve spike zamanlarında bit eşdeğer değildir. Her ikisi kendi bağımsız tamsayı referansına karşı sınanır.

## 7. Tamamlanan yerel doğrulama

| Panel | Kanal | Akış | Kod sözcüğü | Dense başarısız | Gate başarısız | Atlanan rezonatör güncellemesi |
|---|---|---:|---:|---:|---:|---:|
| Regresyon | fixed | 96 | 1038 | 28 | 28 | %65.66 |
| Regresyon | random | 96 | 1030 | 105 | 105 | %64.12 |
| Yeni tohumlar | fixed | 48 | 518 | 13 | 13 | %65.44 |
| Yeni tohumlar | random | 48 | 521 | 47 | 47 | %62.87 |

Toplam **288 akış / 3.107 kod sözcüğü**. Bu panelde farklı uzman kararı, yeni paket başarısızlığı veya tespit edilmemiş yanlış CRC görülmedi. 228 rezonatör sayaç hücresi ve 304 spike zamanı farklıydı. Materyalize güncelleme anlarında gözlenen en yüksek durum farkı 8 Q20 LSB oldu. Bu sonlu deney genel bir kayıpsızlık veya istatistiksel non-inferiority kanıtı değildir.

192 akış önceki panelin regresyon kontrolüdür; 96 akış yeni tohumlarla üretilmiştir. Modeller dondurulmuştur. Ön uç sabit noktalı; sonraki uzman/decoder hâlâ mevcut CPU referansıdır. Giriş her iki kola aynı şekilde kuantize/kırpılmıştır. Sonuçlar önceki float deneyin BLER rakamlarıyla gate iyileştirmesi olarak kıyaslanmamalıdır.

Ek olarak:

- 14 vektör / 1.208.000 örnekte native C++ dense ve gate çıktıları ayrı Python tamsayı ön uç referansıyla eşleşti.
- Dört kopyalı native C++ sürümünde seçilmiş `rtl_smoke` ve `fixed_narrowband` testleri geçti; `rtl_smoke` RUN/VERIFY tekrar kontrolünü içerir.
- FFT beklenen çıktıları mevcut native C++ uygulamasından gelir; bağımsız Python FFT doğrulaması bu pakette yapılmadı. Gerçek `ap_int`, RTL ve kart karşılaştırmaları derleme/çalıştırma aşamasında zorunludur.
- 8 çevrimdışı ölçüm testi geçti: enerji birimi, hız reddi, kalibrasyon, yanlış IP reddi, kopya bölme ve tekrar bazlı güven aralıkları.

## 8. Enerji hesabı ve yorum sınırı

Doğru dönüşüm:

`E [nJ/örnek] = P [mW] / hız [MS/s]`

Örneğin 10 mW / 1 MS/s = 10 nJ/örnek. Ek bir `/1000` uygulanmaz. Eski `reference/measure.py` bu mutlak birimde hata içeriyordu; yeni runner düzeltilmiş hesabı kullanır.

Her mod için üç ayrı kapsam saklanır:

1. Ölçülen toplam SOM giriş gücü / gerçekleşen hız: tüm ölçüm düzeneğinin enerjisi.
2. Aktif güçten iki komşu idle bloğunun ortalaması çıkarılmış enerji: idle'a göre artış.
3. Aynı tekrardaki D0 enerjisi çıkarılıp dört kopyaya bölünmüş fark: **operasyonel, kopya başına tahmin**.

Negatif farklar negatif fiziksel enerji olarak sunulmaz. Kontrol, saat ağı, LUT, PS/sensör okuma ve bekleme etkileri ölçümde bulunur. Eski aritmetik meşgul bekleme yerine `ap_wait()` tabanlı bekleme kullanılmıştır; kontrol sayacı/saat maliyeti yine sıfır varsayılmaz. D0 çıkarımı saf bir bileşen enerjisini kusursuz izole etmez.

%95 nominal Student-t aralıkları sensörün binlerce ham örneğinden değil, **eşleştirilmiş tekrar ortalamalarından** hesaplanır. Varsayılan beş tekrar küçük bir örneklemdir; vektörler arası sonuçlar ayrı raporlanır. FFT karşılaştırmaları keşifseldir; çoklu karşılaştırma düzeltilmiş genel üstünlük iddiası yoktur.

Aynı bitstream karşılaştırması gate'in kontrollü etkisini ölçer; gate donanımı dense modda da fiziksel olarak yer kaplar. Tek başına dağıtılacak dense ve gated devrelerin mutlak maliyeti için daha sonra ayrı optimize bitstream'ler ve aynı kalite/hız kapsamı gerekir. **Eski “300×” analitik hesabı ile bu fiziksel ölçümden doğrudan bir oran zinciri kurulmaz.**

## 9. Tekrarlanabilirlik ve dosyalar

- `evidence/fixed_quality_summary.json`, `quality_shards`: 288 akışın sonuçları.
- `evidence/native_verification.json`: Python/C++ ön uç karşılaştırması.
- `hls/src`: gerçek HLS adayı, gate LUT ve kaynak kimliği.
- `vectors/manifest.json`: giriş ve beklenen çıktı SHA-256 kayıtları.
- `build_kv260.py`: bilgisayardaki tek giriş noktası.
- `board/run_gate.py`: kart doğrulaması, hız kalibrasyonu, güç ve analiz.
- `reference`: önceki kaynakların tarihsel kopyası; buradaki eski scriptleri çalıştırmayın.
- `software`: dondurulmuş model ve CPU referansı. Model pickle'ları yalnızca bu paketin sağladığı dosyalardır.
- `PACKAGE_SHA256.json`: teslim dosyalarının bütünlük listesi.

İsteğe bağlı yazılım yeniden doğrulaması; kartta veya ilk donanım derlemesinden önce zorunlu değildir:

```bash
python -m pip install -r requirements-quality.txt
python validate_fixed.py --workers 4
python prepare_vectors.py
python -m unittest discover -s tests -v
```

Yerel ortam Python 3.12, NumPy 2.2.6, SciPy 1.17.0, scikit-learn 1.8.0 ve Numba 0.61.2 idi. `software/vendor/frozen_v4/requirements.txt` tarihsel sürümdür; yeni kalite tekrarında üstteki `requirements-quality.txt` kullanılır. Kaynak geliştiricileri değişiklikten sonra `refresh_identity.py` ile eski build kaydını geçersizleştirmeli ve ilgili doğrulamaları yeniden yapmalıdır.

## AMD birincil başvuruları

- [Vitis HLS C simulation, 2025.2](https://docs.amd.com/r/2025.2-English/ug1399-vitis-hls/Running-C-Simulation)
- [Vitis HLS protocol / explicit wait](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Protocol)
- [Vitis HLS C/RTL co-simulation configuration](https://docs.amd.com/r/2025.1-English/ug1399-vitis-hls/C/RTL-Co-Simulation-Configuration)

Bu belgeler kullanılan araç akışının kaynağıdır; bu yeni tasarımın sentez başarısı veya güç kazancı için kanıt değildir.
