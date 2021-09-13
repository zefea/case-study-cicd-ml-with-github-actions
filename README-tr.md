
[English](./README.md)

# Case Study CI/CD ML With Github Actions


### İçerik Tablosu


<!--ts-->
   * [Giriş](#Giriş)
   * [Gereksinimler](#Gereksinimler)
      
  
   * [Tensorflow ile Image Classification Modeli Oluşturmak](#Tensorflow-ile-Image-Classification-Modeli-Oluşturmak)
   * [Modelin Containerlaştırılma](#Modelin-Containerlaştırılma)
      * [Docker kurulumu](#Docker-kurulumu)
      * [Dockerfile](#dockerfile)
      * [Image oluşturulması](#Image-oluşturulması)
      * [Container oluşumu](#Container-oluşumu)
      * [Container içerisinde eğitim ve test](#Container-içerisinde-eğitim-ve-test)
      * [Container ve image silinmesi](#Container-ve-image-silinmesi)
   * [ML Süreçleri İçin GitHub Actions ile CI/CD Pipelinelarının Oluşturulması](#ML-Süreçleri-İçin-GitHub-Actions-ile-CI/CD-Pipelinelarının-Oluşturulması)
      * [action.yml dosyasının oluşturulması](#action-.-yml-dosyasının-oluşturulması)
      * [Bir hareket başlatılması](#Bir-hareket-başlatılması)
      * [Başarı durumu](#Başarı-durumu)
      * [Sonuçların incelenip, pull request değerlendirmesi](#Sonuçların-incelenip,-pull-request-değerlendirmesi)
   * [Sonuç](#Sonuç)

<!--te-->



<a name="Giriş"></a>
## Giriş
Bu projedeki motivasyon, çelik üretiminde ortaya çıkan hataları tespit eden ML modelinin, CI/CD yapısı ile birlikte geliştirilmesi, uyarlanması ve
GitHub reposu her güncellendiğinde buradaki modelin otomatik şekilde tetiklenen pipelinelar ile test edilmesidir.

Buradaki temel amaç, farklı fikirler ile geliştirilen birden fazla model arasından, üretim için en iyisinin hangisi olduğuna karar verme sürecinin hızlı ve kolay bir şekilde yönetilmesidir. GitHub Actions, bu durum için kullanılabilecek en iyi yöntemlerden biridir.

Bu çalışmada, ilk kısım var olan çelik __resimlerinin sınıflandırılması__, Tensorflow kullanılarak model nasıl oluşturulur, eğitilir ve test edilir gibi sorular yanıtlanır. İkinci kısımda ise amaç __kontayner(container)__ kavramına ve işleyişi hakkında bilgi sahibi olunmasıdır. Son kısımda ise __CI/CD Pipelineların oluşturulmasıyla__ proje tamamlanmış olur. Proje sonunda modelin eğitimi, test edilmesi otomatik bir şekilde gerçekleşip raporlanır ve rapora göre de pull request kararı alınabilinir.



<a name="Gereksinimler"></a>
## Gereksinimler
-	GitHub
-	Git 
-	Python
-	Tensorflow
-	requirements.txt dosyasındaki kütüphaneler
-	Docker



<a name="Tensorflow ile Image Classification Modeli Oluşturmak"></a>
## Tensorflow ile Image Classification Modeli Oluşturmak
* Dosyalar: Dataset/ NEU-CLS-64_tf_mode, train.py, test.py

Bu kısımda elde olan çelik resim verilerinin sınıflandırılmasıyla ilgilidr. Temel bir makine öğrenimi iş akışını(workflow) takip eder:
   1. Verileri inceleyin ve anlayın
   2. Bir giriş ardışık düzeni oluşturun
   3. Modeli oluşturun
   4. Modeli eğit
   5. Modeli test edin
   6. Modeli geliştirin ve işlemi tekrarlayın

Sınıflandırılma aşaması için, Tensorflow'un bilgisayara kurulumu pek kolay olmayabilir. Dolayısıyla vakit kaybı yaşamayıp, kodun çalışıp çalışmadığını görmek için Google Colab kullanılabilinir. Daha sonraki aşamalarda container ve GitHub actions içerisinde gerekli olan her şey hazır olarak kurulu olucaktır.   

İlk olarak dataseti inceleyelim. İçerisinde eğitim(train), doğrulama(validation) ve test için, 6 farklı klasörde, toplam 4200 çelik resmi bulunmaktadır.

![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/03cacb96becc4a3e8cfb119d5b2c9946adad92bc/images/ss29.png)

* Dosya: 
  - train --> eğitim için (training)
  - test --> doğrulama resimleri (validation)
  - finalTest --> test edilecek resimler (daha önce görülmemiş yeni resimler)


Modelin, birçok katmanla birlikte 2 CNN katmanı vardır. Modeli incelemek için, `model.summary()` :  

   ``` 
   Found 4200 files belonging to 6 classes.
   Found 1048 files belonging to 6 classes.
   ['cr', 'in', 'pa', 'ps', 'rs', 'sc']
   Model: "sequential"
   _________________________________________________________________
   Layer (type)                 Output Shape              Param #   
   =================================================================
   rescaling (Rescaling)        (None, 64, 64, 3)         0         
   _________________________________________________________________
   conv2d (Conv2D)              (None, 64, 64, 16)        448       
   _________________________________________________________________
   max_pooling2d (MaxPooling2D) (None, 32, 32, 16)        0         
   _________________________________________________________________
   conv2d_1 (Conv2D)            (None, 32, 32, 32)        4640      
   _________________________________________________________________
   max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
   _________________________________________________________________
   dropout (Dropout)            (None, 16, 16, 32)        0         
   _________________________________________________________________
   flatten (Flatten)            (None, 8192)              0         
   _________________________________________________________________
   dense (Dense)                (None, 128)               1048704   
   _________________________________________________________________
   dense_1 (Dense)              (None, 512)               66048     
   _________________________________________________________________
   dense_2 (Dense)              (None, 6)                 3078      
   =================================================================
   Total params: 1,122,918
   Trainable params: 1,122,918
   Non-trainable params: 0
   ```


   ### Outputs 

   Dosyalar: model artifact, training-validation.png, metrics.json

   -  Eğitilen model, `model-artifact` dosyasına kaydedilir. Aynı şekilde eğitim ve doğrulama sonuçları grafiği oluşturulup `training-validation.png` olarak kaydedilir.
   -  `Model-artifact` dosyasındaki modelin test edilmesiyle elde edilen doğruluk yüzdeleri `metrcis.json` dosyasına yazılır. 

   ```json
   {
      "accuracy": 0.8170498013496399,
      "Loss": 0.48638322949409485 
   } 
   ```


<a name="Modelin Containerlaştırılma"></a>
## Modelin Containerlaştırılma

* Eklenen yeni dosyalar: Dockerfile, requirements.txt

train.py dosyasında oluşturulan modelin docker ile container haline getirilebilmesi için öncelikle dockerın bilgisayara kurulması gerekmektedir. 
#### Docker kurulumu:  
   * https://docs.docker.com/engine/install/


### Dockerfile
Container oluşturabilmesi için özellikleri (image) belirleyen dosyadır. Bu projede oluşturulan containerın içerisinde tensorflow ve requirements.txt dosyasında bulunan kütüphaneler (numpy vb.) hazır bulunur. Bunlarla birlikte “case-study-cicd-ml-jenkins” dosyası içerisinde (working directory) üzerinde çalışılan train.py, test.py, dataset, docs dosyaları bulunmaktadır. Dockerfile ile bir image oluşturulmaya başlandığında, bütün bu özellikler kurulur, train.py dosyası run edilir ve model oluşur. 


### Image oluşturulması

- Image için verilen isim: docker-model
```sh
   $ sudo docker build -t docker-model -f Dockerfile .
```

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss1.png)

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss2.png)
 
  Dockerfile içerisindeki her adım tamamlanmıştır. Model eğitilip, kaydedilmiştir. 


* Var olan ve oluşturulan bütün imageleri görmek için:
```sh
  $ sudo docker images
```
   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss4.png)

   Yukarıdaki resimde de görüldüğü gibi Dockerfile ile oluşturulan `docker-model` image listesinde bulunmaktadır. 


### Container oluşumu

* Oluşturulan image aşağıdaki gibi run edildiğinde tek seferlik bir container oluşur. Ve image oluşturulurken eğitilen model, test edilir. 
```sh 
  $ sudo docker run docker-model python3 test.py
```

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss5.png)


* Container içerisine girip birden fazla işlem yapılabilinir. (Ör: eğitim ve test tekrarı) 
```sh
  $ sudo docker run -i -t -p 8080:80 docker-model
```

* Container içerisinden çıkıldığında (çalışması bittiğinde) otomatik silinmesi isteniliyor ise `--rm` eklenir. 
```sh
  $ sudo docker run --rm -i -t -p 8080:80 docker-model
```
   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss6.png)

   * Yukarıdaki resimde container içerisine girilmiştir, resimden çıkarılabilecek bazı bilgiler: 


      - root@31d534d13a4b:/case-study-cicd-ml-with-jenkins# 
         - Container id: 31d534d13a4b
         - Working directory : case-study-cicd-ml-with-jenkins
         - Working directory içerisinde `ls` komutuyla görülen dosyalar: 

               dataset   docs  'model artifact'   requirements.txt   test.py   train.py

 
* Container içerisinden çıkış yapmak için `ctrl+d` basılır ve exit yazısı görülür.  


* Çalışır durumda olan olan container listesini görebilmek için:  
```sh
  $ sudo docker container ls
```

* Var olan tüm (çalışır durumda ya da durmuş) containerların listesini görebilmek için: 
  ( `-rm` ile çalıştırıldıysa silinmiştir.) 
```sh
  $ sudo docker container ls -a 
```

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss9.png)


### Container içerisinde eğitim ve test

Container içerisine girildikten sonra, basitçe olması gereken tüm gereklilikleri yüklenmiş, kurulmuş bir bilgisayar olarak hayal edilebilinir. 

* Örneğin `python3 train.py` komutu ile tekrar eğitim yapılabilinir ve  `python3 test.py` ile test edilebilinir. 


  ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss7.png)
  ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss8.png)


### Container ve image silinmesi

İmage her run edildiğinde yeni bir container oluştuğu için, iş bittikten sonra container çalışması durdurulup silinmesi gerekir. 
Container ve image, isim veya id numarası ile silinir. 

* Container durdurulması:  
```sh
  $ sudo docker container stop container-id
```

* Container silinmesi:
```sh 
  $ sudo docker rm container-id
```

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss10.png)


* Image silinmesi: 
```sh
  $ sudo docker rmi model-name
```

Not: Var olan bir image, onu kullanan bütün containerlar kaldırılana kadar silinemez. Aynı image başka container tarafından kullanılıyor olabilir. Dolayısıyla öncelikle oluşturulan container silinmelidir. Bu tip bir problemle karşılaşmamak için container oluşturulurken (image run edilirken) `--rm` komutu eklenmelidir. `--rm`, iş bittikten sonra containerı otomatik siler.



<a name="ML Süreçleri İçin GitHub Actions ile CI/CD Pipelinelarının Oluşturulması"></a>
## ML Süreçleri İçin GitHub Actions ile CI/CD Pipelinelarının Oluşturulması
Bu kısımda, ilk olarak pipelineların kurulması gerekmektedir. Pipeline, sıralı olayların ya da işlerin, teker teker çalıştırılmasına denir. Bu projede, önce kodun kalite kontrolünden geçişi, daha sonra eğer yeterince iyiyse, sırayla model oluşumu, eğitimi ve test edilmesi ve outputların kaydedilmesi gelir. Son olarak ise raporlanmasıdır. Bu durum her repository içerisine `push` yapıldığında tekrarlanır. 


### action.yml dosyasının oluşturulması

   Github Actions içerisinde workflow oluşturmak için, seçeneklerden `Actions` seçilip bir taslak seçilir veya yeni dosya ekle kısmından gerekli path oluşturularak da yapılabilinir.

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss12.png)

   Olması gereken path: ".github/workflows/name.yml"

   İşletim sistemi olarak ubuntu son versiyon ayarlanmıştır. Ve tüm gerekli diğer kurulumlar yapılmıştır. 
   ```sh
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - uses: iterative/setup-cml@v1  
      
      - name: Install requirements for training
        run: pip install --quiet --requirement requirements.txt
   ```
   İkinci kısımda oluşturulan container burada kullanılabilinir. Fakat tüm gerekli özellikler kolayca bulunup kurulabildiği için, kullanılmaması tercih edilmiştir.


### Bir hareket başlatılması

   Pipeline hazırlığı tamamlandığına göre, repoda bir değişiklik yapılırsa(push) action otomatik olarak başlayacaktır. İlk olarak python kodları kalite kontrolünden geçip, sonra eğitim ve test kısmı başlatılacaktır. 

   ```sh
     $ git add .
     $ git commit -m "branch is updated with a commit"
     $ git push origin feature/ci-cd
   ```

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss18.png)

   `pull request` sayfasından görüldüğü gibi kalite kontrol tamamlanmış ,eğitim başlamıştır. 

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss22.png)

   Aynı sonuçlar `actions` kısmından da görülür, action takibi buradan daha detaylı yapilabinir.

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss20.png)


### Başarı durumu

   * Eğer her şey sorunsuz bir şekilde tamamlanmışsa, yeşil işaretler ortaya çıkar ve artık merge yapılabilir.  
   
     ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss25.png)

   * Son olarak, eğitim sürecinde oluşan model ve test sonuçları `my-artifact` dosyasına kaydedilir. 

     ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss16.png)


### Sonuçların incelenip, pull request değerlendirmesi

   Eğitim ve test sonuçlarının raporu `pull request` sayfasından görülebilinir. 
   
   * Buradaki raporu incelersek eğer, %80 doğruluk oranına sahip olan bu model için çok başarılı bir eğitim gerçekleştirilememiştir. Grafikten yorumlamak gerekirse, train ve validation çizgilerinin aynı anda azalıp artması demek test sonucunun başarılı olucağını gösterir. Fakat bu grafikte bu durum yeterince oluşmamıştır. Dolayısıyla bu model eğer `develop` branchindeki modelden iyi sonuç veremediyse pull request kabul edilmez. 

     ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss23.png)

     ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss24.png)



<a name="Sonuç"></a>
## Sonuç

   Her bir kişi kendi `feature branch` içerisinde model geliştirip eğitip test edebilir. Ve her `pull request` yapıldığında bu modelin sonuçları raporlanıp sunulup, oluşturulan modelin daha iyi olup olmadığına karar verilir. Bu karar olumluysa `develop` branchine taşınır, proje sonunda da asıl model olarak seçilip `main` branchine konulur. Bunların hepsi otomatik gerçekleşir ve böylece proje yönetimi kolay ve hızlı bir şekilde sağlanmış olur.  

   Bu projede eğitim de test de Github actions üzerinden gerçekleşiyor. Fakat bazen eğitimi kişinin kendi sisteminde yapıp burada sadece önceden oluşturduğu modeli test etmesi işlerin daha da hızlı olmasını sağlayabilir. 


<a name="Kaynaklar"></a>
## Kaynaklar

https://www.tensorflow.org/tutorials/images/classification
https://www.tensorflow.org/tutorials/keras/classification
https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/16_cnn_cifar10_small_image_classification/cnn_cifar10_dataset.ipynb
https://xaviervasques.medium.com/quick-install-and-first-use-of-docker-327e88ef88c7
https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f
https://aws.amazon.com/blogs/opensource/why-use-docker-containers-for-machine-learning-development/
https://towardsdatascience.com/have-your-ml-models-built-automatically-using-github-actions-5caa03c6f571
https://towardsdatascience.com/from-devops-to-mlops-integrate-machine-learning-models-using-jenkins-and-docker-79034dbedf1 
https://www.youtube.com/playlist?list=PL7WG7YrwYcnDBDuCkFbcyjnZQrdskFsBz 
