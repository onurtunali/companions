# 1. Neden Ölçeklenebilir Hesaplama Önemli

Verisetlerinin sınıflandırılması:

| Veriset Tipi | Boyut Aralığı | RAM'e sığıyor mu | Diske sığıyor mu|
| --- | --- | --- | --- |
| Küçük | 2 - 4GB arası | Evet | Evet |
| Orta | 2TB altında | Hayır | Evet |
| Büyük | 2TB üstünde | Hayır | Hayır |

Dask yapısı

<img src="fig/dask_api.png" width=400>

**scaling up:** Daha güçlü makine ekleme sınırlı bir yerden sonra daha iyi makine eklemek için AR&GE yapıp daha güçlü makineyi üretmek gerekebilir!

**scaling out:** Daha fazla makine ekleme

Dask çalışırken iki türlü problemle karşılaşabilir. worker hatası, veri hatası. Worker hatasından başka bir worke işi devralarak kalan görevi bitirebilir. Fakat veri hatasından worker'ın şimdiye kadar yaptıkları kaybolur ve baştan başlanması gerekir.

# 2. Dask'a Giriş

Dask sütun değerlerinden örneklem alarak sütunların veri türlerini belirler. Bu yüzden binlerce verinin arasından sadece bir tanesi'nin (`int` yerin `Nan` değerlerin bulunması) uyumsuzluğu hesaplamalar sırasından hata üretir. Veri türunun açıkça belirlendiği `parquet` gibi dosya türleri ile aşılabilir.


# 3. Dask `DataFrame`'lere Giriş

Dask DataFrame olarak pandas kütüphanesini kullanır ve bölümlediği (*partition*) her alt parça bir pandas DataFrame'den oluşur. Bu yüzden pandas metotlarının çoğu kodlanmıştır.

Dask `DataFrame`'ler herhangi bir filtreden geçirildikten sonra bölümlerdeki satır sayısı dengesizleşebilir ve `repartition` metotu çağırılarak dengesizlik düzeltilebilir.

Ekleme çıkarma gibi işlemler Dask `DataFrame`'ler için gerçekleştirilemez çünkü bu veri yapıları sabittir (*immutable*). Aynı şekilde `join/merge` işlemleri de veri shuffle gerekliliğinden dolayı daha maliyetlidir. Hızlıca gerçekleştirmek için verideki benzersiz sütunlardan biri indeks olarak kullanılabilir ve `join/merge` işlemleri bunlar üzerinden gerçekleştirilebilir.

# 4. `DataFrame`'lere Veri Yükleme

Büyük veriyi saklamak için `parquet` formatı `csv`'ye göre çok daha verimlidir. Hem depolama boyutu hem de işlem hızı açısından 5x kadar performans kazancı sağlanır.

# 6. Summarizing and Analyzing `DataFrame`'s

Veri Bilimi'nin %80 veri temizleme ve önişlemedir.

## Betimsel İstatistikler

Pozitif (sağ) çarpık (skew)'da ortalamanın altındaki değerler daha olasıdır.
Negatif (sol) çarpık (skew)'da ortalamanın üsütündeki değerler daha olasıdır.

Groupby operasyonları partisyon oluşturmada baz alınmış sütunlarda yapılmalı. Aksi takdirde shuffe işlemleri operasyonu çok yavaşlatır.