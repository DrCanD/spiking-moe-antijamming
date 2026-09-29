UAV Jamming v4 — RS+CRC kodlu sürekli akış

Colab için teslim edilen UAV_Jamming_v4_Tek_Hucre.py dosyasının tamamını bir
hücreye yapıştırın. CPU ve varsayılan WORKERS=2 yeterlidir; GPU kullanılmaz.
Varsayılan pipeline smoke -> development -> seçim kilidi -> overnight şeklindedir.

Bilimsel kapsam

RS(255,159): 155 uygulama byte + 4 CRC-32 byte. 96 parity byte. Her bağımsız
akışta ilk kod sözcüğünün başlangıcı 0..2039 sembol aralığında rastgeledir.
Kod sözcükleri 5 ms karar bloklarıyla hizalanmaz. İlk sözcükler atılmaz;
prefix/tail filler bitleri goodput paydasına dahildir. CRC'yi geçen yanlış
payload'lar ayrıca sayılır; yalnız truth ile doğrulanmış doğru uygulama bitleri
yararlı veri kredisi alır. Truth alıcıya ve decoder'a verilmez.

Model/ön uç/ALE durumu akış boyunca korunur, paket sınırında sıfırlanmaz.
Yönlendirici blok t sonunda karar verir, t+1 bloğunda uygular. Maske aynı 5 ms
tampon içindeki RMS'e bakabilir; gelecekteki bloklara bakamaz. block_refine2,
v3'ün 100 ms frame refinement'ı değildir. Bu fark ayrı isimle belirtilmiştir.
RS sözcüğü 20.4 ms'de gelir; en fazla ek 5 ms blok bekleme gerekir. Decoder
CPU süresi ve gerçek cihaz gecikmesi bu teorik kullanılabilirlikten ayrıdır.
İdeal sembol ve paket senkronizasyonu varsayılır; interleaving/ARQ yoktur.

Beş ön uç: spike, analog_delta, analog_raw, delta, FFT. Aynı eğitim dalgaları,
RF derinliği/ağaç sayısı ve koşulları paylaşılır. analog_raw ilk beş delta
özelliğini de içerir. Saf SNN veya nöromorfik donanım dağıtımı iddiası yoktur.
Kural kontrolü her blokta sürekli hızlı ALE denemesi yapar, enerji oranıyla
çıktıyı seçer ve pulse maskesi uygular. Kural/ham kontroller sınıflandırıcı
kullanmaz. Ön uçlar örneklenen karar sıklığından bağımsız her blokta çalışır.

Seçim

Önce guarded-periodic DEV BLER ile maske seçilir. Ardından her ön uç x koşul
x senaryo hücresinde, aynı maskeli guarded-periodic'e göre BLER en fazla
1 yüzde puan yüksek; doğru korunan bit oranı en fazla 1 yüzde puan düşük;
ilk NB bölümündeki doğru korunan oran en fazla 2 yüzde puan düşük olabilir.
Ek CRC kabul edilmiş yanlış teslimat da olmamalıdır. Her hücrenin geçmesi
gerekir. Uygun adaylar arasında en az RF çağrısı seçilir. Hiçbir seyrek aday
uygun değilse periyodik seçilir. Bu bir seçim kuralıdır; istatistiksel
noninferiority/eşdeğerlik kanıtı değildir. Overnight seçimde kullanılmaz.

Adaylar: v3 guard8, guard4, guard2, confirm8 ve guarded-periodic. confirm8,
eylem değiştiğinde, güven <0.75 olduğunda veya değişim tetiklendiğinde sonraki
iki blokta tekrar karar ister; gerekirse istek yenilenir. v3 guardsız legacy8
mekanizma kontrolü olarak saklanır. Eşikler protocol.json içinde sabittir.

DEV: 3 model tohumu x 2 koşul x 5 senaryo x 12 akış = 360 akış.
TEST: 5 yeni model tohumu x 2 koşul x 8 senaryo x 40 akış = 3200 akış.
Eğitim her model/koşul için 5 sınıf x 20 tek-jammer akışı; her akışta bir
ısınma ve beş eğitim bloğu. Karışımlar ve geçişler eğitime verilmez.
DEV senaryoları clean, nominal, short_dwell, low_snr, strong_pulse;
TEST'e long_dwell, weak_pulse, high_duty da eklenir. Model hep SNR 10 eğitim
verisini kullanır; düşük SNR ve yeni duty/dwell koşulları ayrı raporlanır.

Çalıştırma, kesilme ve dosyalar

Paket kendi Python venv'ini oluşturur, sabit requirements.txt sürümlerini kurar.
Smoke ve algoritmik denetimler geçmeden uzun koşu başlamaz. Sonuçlar yeni v4
Drive klasörüne her iki akıştan sonra atomik parçalar halinde kaydedilir.
Aynı hücre tekrar çalıştırılırsa doğrulanmış parçalar ve modeller atlanır.
Aynı çıktı klasörünü iki Colab oturumunda eşzamanlı çalıştırmayın. Kaynak,
Python sürümü, dependency veya seçim değişirse mevcut klasöre devam edilmez.
Çalışma süresi ve Colab oturumunun tamamlanması garanti edilmez.

Tüm pipeline tamamlanınca ana klasörde RESULTS_REVIEW.zip oluşur. Gelişim,
seçim ve test kanıtları birlikte içerilir. Modeller Drive'da saklanır; sonuç
ZIP'inde yalnız model metadata/checksum kayıtları vardır. Paket doğrulaması,
alınan waveform veya pickle byte'larını yeniden üretmekle aynı değildir.
CSV tabloları: links, per_seed, word_positions, selection_gates (DEV),
paired_contrasts (TEST). Ham parçalar CRC ve sözcük bazında bütün sayımları taşır.

Yerel kullanım (ayrı sanal ortamda):
python -m pip install -r requirements.txt
python verify_v4.py
python run_v4.py --profile smoke --output /path/smoke --workers 2
python run_v4.py --profile development --output /path/development --workers 2
python run_v4.py --profile overnight --output /path/overnight --workers 2 --selection /path/development/selection.json

Donanım enerji iddiası yoktur. Yazılım ön uç özelliklerini ve birebir aynı
decoder girdilerini hız için paylaşır; bu paylaşım yöntemler arasında truth
veya karar taşımaz. Duvar süreleri seri politika döngüsü, RF predict ve uzman
işlemleri için kapsamı belirtilmiş simülasyon ölçüleridir; cihaz başına enerji
veya uçtan uca gerçek zamanlı kapasite olarak yorumlanmaz.
