# PROMETHEUS

Prometheus bir izleme (monitoring) sistemidir. Zaman serilerini belirli aralıklarla scrape eder ve depolar. Veri modeli zaman serileri ve key-value pair şeklindeki boyutlardan oluşur. Bir zaman serisi için genel syntax:

times_series_name{key="value"} # Farklı keyler'e label denir.

Scrape interval'ın 15sn olması tüm metriklerin aynı zamanda toplanacağı anlamına gelmez. Prometheus buradaki yükü dağıtarak metrikleri toplar ve aynı zaman aralığında gelen metrik sonuçlarının timestamp'leri farklı olabilir.

# İsimlendirme En iyi Pratikler

- Counter metrikleri `_total` son eki ile bitmelidir.
- Metrikler birimlerden bağımsız float (64 bit) veri tipi şeklinde tutulur. Bu yüzden metriğin standart birimi isminde geçmelidir. Örneğin istek işleme sürelini tutan bir metriğin birimi `_seconds_` şeklinde metrik ismine eklenmeliri.
- Eğer iki veya daha fazla farklı metrik sürekli birleştirilerek (aggregate) kullanılıyorsa bunlar bir label altında toplanmalıdır. Örneğin farklı iki endpointe giden istekleri `metricname_login_total` ve `metricname_subscribe_total` şeklinde isimlendirip, toplam istek sayısını bulmak için sürekli bunları toplamak yerine, `metricname_requests_total{path="/subscribe|/login"}` şeklinde metriklendirme yapılmalıdır.

# Zaman serisi çeşitleri

İki tür zaman serisi vektörü vardır:

1. Instant vektör: Sorgulandığı timestamp anındaki tek değer.
2. range vektör: Sorgulandığı zaman aralığındaki tüm değerler. Örneğin zaman aralığı [1m] şeklinde verilirse, sonuç olarak gelen değerler sorgu anındaki timestamp - 1m aralığındaki tüm değerleri getirir.

# Grafana Time Range

Grafana için grafik time range dashboardda seçtiğimiz aralıktır. Genellikle saniyeye döndürülür. Grafana verilen PromQL sorgusunu ekranda grafik oluşturmak için bu grafik time range / min interval kadar çalıştırır ve bunların sonuçlarından da grafik çizer. Eğer belirlenen interval sonucu gelecek sonuçlar çizilebilecekten çok daha fazla ise otomatik olarak min interval tanımlanır. 

Örneğin çözünürlüğün 1000 pixel, grafik time range'in 10 saat ve zaman aralığının da 10sn şeklinde verildiğini kabul edelim. 10 saat, 10 * 60 * 60 = 36.000 saniyeye denk gelir. 36.000 / 10 = 3600 adet sonuca denk geldiği ve 1000 pixel'den büyük olduğu için Grafan bu sonuçlara gösterecek çözünürlüğe sahip değildir. Bu durumların önüne geçmek için Grafan 36.000 / 1000 = 36 şeklinde bir minimum zaman aralığı belirler ve 10sn yerine bu 36sn'yi yuvarlayarak bir sayı kullanır.

# Label Matchers

- **=:** Equal `job="node"`
- **!=:** Not equal `job!="node"`
- **=~:** Reger `job=~"node.*"`


# ETC

**exporter:** Deploy edilmiş bir sistemin yanında çalıştırılan ve sistem metriklerini ifşa eden yazılımlardır. Örneğin node_exporter çalışdığı donanımın cpu, memory, load gibi metriklerini ifşa eder. 

**sum:** Bir metriğin farklı dimension(label)larını toplamak için kullanılır. Zamanla veya farklı timestampteki değerlerle bir ilgisi yoktur.
