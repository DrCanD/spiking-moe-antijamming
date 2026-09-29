# UAV Jamming v5: blok kararı ve kısa ALE kurtarması

Bu paket v4 kaynaklarını `frozen_v4/` altında değişmeden tutar. V4 sonuçları görüldüğü için yeni mekanizma geliştirme adayıdır; v4 sonuçları yeniden doğrulama testi olarak kullanılmaz.

Amaç: ilk paket ve karıştırıcı geçişlerindeki kaybın ne kadarının bir blok gecikmeli karardan geldiğini ayırmak; alınan sinyalden tetiklenen sınırlı hızlı ALE kurtarmasını sınamak.

Her ön uç için 7 politika, ayrıca rule ve raw kontrolleri vardır: toplam 16 yöntem. Ön uçlar spike ve analog_delta; maskenin tüm yöntemlerde tek seçimi block_refine2. Model eğitimi, RS(255,159), 155 byte yük+CRC32, paket fazı ve temel kanal üretimi v4 ile aynıdır. Geliştirmede maske/threshold araması yapılmaz.

Politikalar:
- lag_periodic / lag_guard8: değiştirilmemiş v4 referansları.
- current_periodic / current_guard8: mevcut tamponlanmış bloktan alınan kararı aynı bloğa uygular.
- refresh_guard8: ucuz risk tetikleyicisi aynı blokta RF yenilemesini zorlar; ALE kurtarması yoktur.
- rescue_guard8: aynı yenilemeye ek olarak risk penceresinde tek kalıcı ALE durumunu hızlı günceller.
- rescue_periodic: aynı kurtarmanın her blok RF kararı verilen kontroldeki etkisini ayırır.

Risk ölçütü: önceki alınan bloğa göre güç oranı >=2 veya <=1/2 ya da normalize lag-1 korelasyon farkı >=0.20. Başlangıçta 2 blok, her tetiklemede tetik bloğu dahil 2 blok risk penceresi. Pencere yeni tetiklerle uzayabilir; bu durumda bütün ek çalışma sayılır. Etiket, gerçek geçiş konumu, paket başlangıcı, doğru bit veya CRC sonucu tetikleyiciye girmez.

Kurtarmada hızlı ALE (mu=0.05) tek kez çalışır. Blok güç azalması rho>0.25 ise artık sinyal, değilse ham sinyal kullanılır. ALE durumu çıktı reddedilse de korunur; yapılan tüm güncellemeler sayılır. Paralel gizli ALE dalı yoktur. Bu eşikler yerel pilot sonuçları görülmeden seçilmiştir; iyileşme garantisi değildir.

Bütün alıcı çıktıları ilgili 5 ms blok tamamlanınca hazırdır. Current politikalar aynı tamponlanmış blok üzerinde işlem yapar; lag politikalar önceki kararın etkisini taşır. Bu blok düzeyinde nedenselliktir. Sıfır gecikmeli örnek düzeyinde alıcı değildir. Kod sözcüğü için 20.4 ms toplama +0..4.99 ms kalan blok beklemesi vardır; hesaplama süresi ayrıca ölçülmelidir.

Sayaçlar: RF çağrıları, ALE örnekleri/tahminleri/güncellemeleri, ALE tap çarpımları, dedektör çarpımları ve kurtarma kabul/reddetmeleri. ALE tap çarpımları taps*(2*predictions+updates) olarak sayılır. Bu kısmi aritmetik iş hesabıdır; toplam MAC veya enerji değildir. Ön uçlar/özdeş çözücü girdileri deney düzeneğinde paylaşılır; zamanlar cihaz enerjisi değildir.

Profiller ve yeni tohum uzayı:
- smoke: 4 akış, yalnız işlev kontrolü; küçük eğitim.
- local_pilot: 3 model tohumu x 2 koşul x 6 senaryo x 3 akış =108. Tam eğitim boyutu, keşif amaçlı; validation seçimi yapamaz.
- development: 5 x 2 x 6 x 12 =720 akış.
- validation: bağımsız 5 x 2 x 6 x 20 =1200 akış. Yalnız development seçim dosyasıyla ve uygun seyrek aday bulunmuşsa açılır.

Senaryolar: clean, nominal, short_dwell, low_snr, strong_pulse, high_duty. Kanal geometrisi v4 ailesindendir; yeni tohumlar yeni fiziksel alan anlamına gelmez. İdeal senkronizasyon/BPSK simülasyonu, interleaver ve ARQ yoktur. Broadband genel başarımı bu panelin kapsamında değildir.

Seçim: current_guard8, refresh_guard8, rescue_guard8 arasından current_periodic referansına göre HER ön uç/koşul/senaryoda BLER +1 yüzde puan, ilk sözcük ve geçişi kesen sözcük BLER +2 puan, doğru tutulan bit -1 puan, ilk NB doğru tutulan bit -2 puan sınırlarını geçen ve ek CRC kabul edilmiş yanlış paket üretmeyen adaylar. Ayrıca ortalama RF çağrısında en az %50 azalma gerekir. Uygun adaylar arasından en az RF çağrısı seçilir. Hiçbiri geçmezse current_periodic fallback; validation açılmaz. Bunlar DEV nokta tahmini ölçütleridir; istatistiksel eşdeğerlik/noninferiority kanıtı değildir. ALE yükü ayrıca raporlanır, RF azalması toplam iş azalması sayılmaz.

Yerel komutlar (çıktı kaynak klasörünün dışında olmalı):
```
python verify_v5.py --output ../verification_v5.json
python run_v5.py --profile local_pilot --output ../v5_local_pilot --workers 2
python run_v5.py --profile development --output ../v5_development --workers 2
python run_v5.py --profile validation --output ../v5_validation --selection ../v5_development/selection.json --workers 2
```

Bağımlılıklar frozen_v4/requirements.txt içinde tam sürümleriyle sabittir. Source/protocol/Python/dependency/seçim kilidi, model ve veri parçası SHA-256 kontrolleri vardır. Aynı komutu aynı kaynak ve ortamla yeniden çalıştırmak tamamlanan parçaları korur. Kaynak/ortam değiştiğinde eski çıktıya karıştırma reddedilir. Bozuk kanıt otomatik silinmez.

Her profil RESULTS_REVIEW.zip üretir. links.csv havuzlanmış BLER/goodput, paired_contrasts.csv ise model sonra bütün akış yeniden örneklenen 2000 tekrarlı mean-stream güven aralıkları içerir. Çoklu karşılaştırma düzeltmesi yoktur. Kod sözcükleri veya yöntem skorları bağımsız akış gibi sayılmaz.

Colab tek hücre başlatıcısında varsayılan MODE='pilot': doğrulama, smoke ve development. MODE='full' ayrıca başarılı DEV seçimi varsa validation çalıştırır. CPU yeterlidir; GPU kullanılmaz. Sonuçlar ayrı v5 Drive klasörüne yazılır, v4 çıktıları değiştirilmez. Aynı hücre aynı build klasöründen devam eder. Kullanıcı başlatıcıyı tekrar çalıştırmadan otomatik arka plan işi yoktur.
