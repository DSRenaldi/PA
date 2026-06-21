~~1. perubahan desain system. dari tahap segmentasi ada garis baru ke tahap sentiment analysis, karena analisis sentiment menggunakan data hasil normalisasi. sedangkan untuk aspect based menggunakan data hasil stemming, sehingga setelah segmentasi tetap berlanjut ke tahap berikutnya hingga stemming sampai ke aspect based. Kemudian setelah dipecah semua digabungkan Kembali yg nantinya akan ketemu di validasi dan visualisasi (done acc)~~

~~2. Tambah data (done acc)~~

~~3. benerin table vector yg ada di buku (done acc)~~

~~4. segmentasi teks tidak dibatasi 2 kalimat saja. Selama ada konjungsi yg didaftarkan maka pecah kalimatnya (done acc)~~

~~5. pada buku jelaskan kenapa menggunakan primary threshold = 0.30, secondary threshold = 0.40, lainnya threshold = 0.18 pada buku (done acc)~~

~~6. tambahkan confussion matrix untuk aspect based dan sentiment analysis di buku (done acc)~~

~~7. develop dashboard (done acc)~~

~~8. tambahkan contoh testing kalimat baru di codenya (done ada di web udh acc)~~

~~9. tambahkan mockup dashboard yg akan dibuat ke dalam buku (done acc)~~

~~10. jadikan 1 file penghapusan slang (done acc)~~

~~11. data train menggunakan data yg non segmented dan data test 100% data(done acc)~~

~~12. tambahkan f1 score dari masing" aspek pada evaluasi aspect based (done acc)~~

~~13. benerin labeling data (done acc)~~

~~14. bikin 2 scenario aspect based, scenario 1 menggunakan data yg tersegmentasi sebagai data test namun hanya data dengan kalimat yg jls saja yg diambil, scenario 2 menggunakan 100% data sebagai data test (done acc)~~

~~15. komentar feedback dihapuskan (done acc)~~

~~16. scrap data baru dengan rentang Waktu 2026 (Data Jan-Mei udh dimerge siap dipakai (done acc))~~

~~17. Judul sub bab tidak boleh miring (italic) meskipun menggunakan bahasa asing (done acc)~~

~~19. cek rule yg sudah didefine untuk  sentiment analysis. Jika ada rule baru maka jabarkan rule barunya di buku (done acc)~~

~~18. Pada website bedakan antara data test dan data train saat menampilkan data (done acc)~~

~~20. Pada website tambahkan halaman untuk user input text baru dan outputnya akan menghasilkan aspek dan juga sentiment. yg dmn nnt sistemnya program dari preprocessing hingga akhir yaitu sentiment analysis diembeded ke dalam aplikasi. Sehingga setelah user input teks atau data, nnt teks atau data tersebut akan diproses sesuai dengan alur yg telah program yg telah dikerjakan. Untuk menyimpan hasil olah input user nnti disediakan file csv kosong untuk menyimpannya. Sistem menyimpan filenya nnt sebelum disimpan akan dilakukan pengecekan terlebih dahulu apakah file csv untuk menyimpan ada atau tidak didalam directory, jika tidak ada akan raise error, jika ada maka cara menyimpan datanya dimerge kedalam file csv agar file lama tidak ter overwrite.(done acc)~~

~~21. Buatkan page yg nntnya bisa untuk menganalisis data secara historical. Namun yg ditampilkan hanya lah data yg memiliki timestamp saja. (done acc)~~

26\. pada karakteristik diceritakan data yg digunakan untuk permodelan menggunakan data yg seperti apa dan data yg digunakan untuk testing aplikasi menggunakan data yg seperti apa (jabarkan secara detail). (done acc)

~~27. Pada bagian parameter Eksperimen yg sebelumnya dua aspek (draft Eval 1), diganti menjadi multi aspek.(done acc)~~

~~28. jelaskan kedua scenario pada (done acc):~~

&#x20;	~~a. 4.1 Parameter Eksperimen~~

&#x09;~~b. 4.6.2 Data Split~~

&#x09;~~c. 4.6.5 Evaluasi Aspect Based~~

~~29. Buatkan design system berbahasa inggris kemudian kirimkan bu entin (tidak masuk buku) (done acc)~~

~~30. aspect based scenario 2 meskipun dia dibandiangkan dengan dirinya sendiri pada data train, akurasinya tidak 100% karena ada pengaruh dari data yg lain. pada analisis hasil eksperimen untuk aspect based scenario 2 ini, jelaskan kenapa terdapat data single aspect yg seharusnya dibandingkan dengan dirinya sendiri yg ada di data train tapi masih bisa salah prediksi. penjelasannya langsung menganalisa data yg salah prediksi. percobaan yg dianalisis adalah percobaan ngram(1,3) (done acc)~~

~~31. yg akan dibahas dibab 2 adalah pdam Surya sembada,komentar keluhan pelanggan PDAM Surya Semabada (done acc)~~

