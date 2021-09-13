
[Türkçe](./README-tr.md)

# Case Study CI/CD ML With Github Actions


### Table of contents

<!--ts-->
   * [Introduction](#introduction)
   * [Requirement Specification](#requirement-specification)
      
  
   * [Image Classification with Tensorflow ](#Image-Classification-with-Tensorflow)
   * [Containerization of The Model](#Containerization-of-The-Model)
      * [Installation of Docker](#installation-of-Docker)
      * [Dockerfile](#dockerfile)
      * [Build the image](#build-the-image)
      * [Run the image](#run-the-image)
      * [Train and test inside a container](#Train-and-test-inside-a-container)
      * [How to delete a container and an image](#How-to-delete-a-container-and-an-image)
   * [Create a Pipeline with GitHub Actions for ML Processes](#Create-a-Pipeline-with-GitHub-Actions-for-ML-Processes)
      * [Setting up action.yml](#Setting-up-action.yml)
      * [Start an action with a commit](#Start-an-action-with-a-commit)
      * [Examine the report in pull request](#Examine-the-report-in-Pull-Request)
      * [Success Status](#Success-Status)
   * [Conclusion-Suggestions](#Conclusion-Suggestions)

<!--te-->


<a name="Introcution"></a>
## Introduction
The motivation in this project is the development and operations of the ML model, which detects errors in steel production, with the CI/CD structure. Every time the GitHub repository is updated, the model will be trained, tested and reported with automatically triggered pipelines. And depending on that report, if the model’s accuracy is satisfactory enough, it will be selected as project model.  
 
The main purpose in this action is to management of the process fast and easily so that it can be decided which model is the best one for the production since there might be several models which is developed by different ideas or people. For this purpose, GitHub Actions is one of the best solutions.

In this study case, first section is __image classification__. Questions like how to create a model, train and test it with Tensorflow will be answered. In the second section, the aim is to be familier with the concept of a __container__. And at the final step, by building the __CI/CD pipelines__, project will be completed. At the end, training and testing can be done automatically and based on the results, decisions about the model can be made using Pull Request.   



<a name="Requirement Specification"></a>
## Requirement Specification
-	GitHub
-	Git 
-	Python
-	Tensorflow
-	Libraries in requirements.txt file
-	Docker



<a name="Image Classification with Tensorflow "></a>
## Image Classification with Tensorflow 
* The files: Dataset/ NEU-CLS-64_tf_mode, train.py, test.py

This part is about how to classify images of steel from given Dataset. It follows a basic machine learning workflow:
1. Examine and understand data
2. Build an input pipeline
3. Build the model
4. Train the model
5. Test the model
6. Improve the model and repeat the process

For this part, rather than working on your own system since Tensorflow is not so easy to be installed, Google Colab is very easy solution to get fast results and see if the code is working or not. it has everything that is needed for now. Because after a while, we will only be needed GitHub to use GitHub actions


Firstly, lets examine the dataset. Dataset has about 4200 photos of steel for training, testing and validation. And each folder contains 6 sub-directories, one per class:

![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/03cacb96becc4a3e8cfb119d5b2c9946adad92bc/images/ss29.png)

* Folder: 
  - train --> for training
  - test --> for validation
  - finalTest --> testing (new images that model did not see before)

The model has 2 CNN layer with other layers. Here is the output of `model.summary()` :  

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

   Files: model artifact, training-validation.png, metrics.json

   - After training, trained model will be saved in a folder named `model artifact` for testing later and a figure about trainig vs validation will be given as an output `training-validation.png` of the model. 
   - After testing, using the `model artifact`, the results will be written in a file called `metrics.json` with accuracy and loss values of it. 

   ```json
   {
      "accuracy": 0.8170498013496399,
      "Loss": 0.48638322949409485 
   } 
   ```



<a name="Containerization of The Model"></a>
## Containerization of The Model

* Additional Files: Dockerfile, requirements.txt

To be able to create and run a container, docker must be installed. 
#### Installation of Docker:  
   * https://docs.docker.com/engine/install/


### Dockerfile
Dockerfile is a text-based script of instructions which tells Docker what will be needed in the application to run (For example, the libraries in requirements.txt). It creates a Docker image. After building the image, container will be ready when image is runned. 
In our Dockerfile, we basically create a folder as working directory which has all needed files like Tensorflow, libraries in requirements.txt, and of course our python scripts with their inputs.


### Build the image

- Name of the image: docker-model
```sh
   $ sudo docker build -t docker-model -f Dockerfile .
```

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss1.png)

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss2.png)
 
   Every step in Dockerfile completed including creating the model and training it.


* To see all images that is exist:
```sh
  $ sudo docker images
```
   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss4.png)

   As we can see from the figure above, image `docker-model` is created from Dockerfile.


### Run the image

* If the builted image runs, the model that is trained before will be tested.
```sh 
  $ sudo docker run docker-model python3 test.py
```

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss5.png)


* Entering inside of a container is possible. (Ex: Training and testing can be done multiple times.)
```sh
  $ sudo docker run -i -t -p 8080:80 docker-model
```

* After getting out of inside of the container, if we want that container to be deleted automatically, `--rm` can be added to the command. 
```sh
  $ sudo docker run --rm -i -t -p 8080:80 docker-model
```
   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss6.png)

   * Above figure shows an inside of a container. Here some information we can get from it: 


      - root@31d534d13a4b:/case-study-cicd-ml-with-jenkins# 
         - Container id: 31d534d13a4b
         - Working directory : case-study-cicd-ml-with-jenkins
         - Inside of the working directory, using `ls` command: 

               dataset   docs  'model artifact'   requirements.txt   test.py   train.py

 
* To get out of container, press `ctrl+d` and "exit" writing will be on the screen. 


* To see the list of a running containers: 
```sh
  $ sudo docker container ls
```

* To see all containers that is exist (running or not):
  (ps: if it is runned with `--rm`, it is deleted if it stoped running)

```sh
  $ sudo docker container ls -a 
```

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss9.png)


### Train and test inside of a container

After entering inside of the container, it is basicly a new computer that has everything you need to run your program. The model can be created and trained as many times as the developer needs. 

* As an example from this project, 
  commands: `python3 train.py` for training, `python3 test.py` for testing can be used.  

  ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss7.png)
  ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss8.png)


### How to delete a container and an image

Every time an image is runned, a new container is created. Because of that, after the process, container must be stopped running and removed. 
Container and images can be deleted either with its name or id number. 

* Stop Container: 
```sh
  $ sudo docker container stop container-id
```

* Remove Container:
```sh 
  $ sudo docker rm container-id
```

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/b1b0deabecf4ebfd07bf04eea3d8ee00544982d8/images/ss10.png)


* Remove image: 
```sh
  $ sudo docker rmi model-name
```

Note: An image can not be removed until every single container which uses that image is removed. Because after a building an image, it can be run several times which means several container is created. So that firstly containers should be removed and then image. Using `--rm` while runing the image, can avoid this kind of problems.   




<a name="Create a Pipeline with GitHub Actions for ML Processes"></a>
## Create a Pipeline with GitHub Actions for ML Processes

In this section, first of all, pipelines should be set up. A pipeline is a sequence of events or jobs that can be executed. The events in this case is lint the code, and if it is good enough, create the model, train and test it, give outputs as artifacts and also give an report about the results of training and testing. Those steps will repeat themselves every time a `push` has made. 


### Setting up action.yml

   To create a workflow in GitHub Actions, you can choose a template from the section of `Actions` in github or you can say add a new file and create your path. 

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss12.png)

   The path is always fixed: ".github/workflows/name.yml"

   Our system runs on ubuntu with a latest version. And other neccesary settings must be done.
   ```sh
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - uses: iterative/setup-cml@v1  
      
      - name: Install requirements for training
        run: pip install --quiet --requirement requirements.txt
   ```

   There is no need to use the container that we created before, because all the neccesaary requirements can be found and set easily. But we could've use the container that has been created before.
 

### Start an action with a commit

   So, now that the pipelines are ready, lets push something in the reporsitory. Action will start automatically. First part is lint for the code. After that the training and testing will start.

   ```sh
     $ git add .
     $ git commit -m "branch is updated with a commit"
     $ git push origin feature/ci-cd
   ```

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss18.png)

   From `pull request`, it can be seen that linting part is done successfully, and training has started. 

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss22.png)

   Also from the `actions` section,the run can be followed: 

   ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss20.png)


### Success Status

   * if everything is okay, there will be a green check sign tells us that program runned without an error and it is okay to merge. 

     ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss25.png)

   * At the end, some of the outputs is saved in a folder named `my-artifact`. Model can be used from it.

     ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss16.png)


### Examine the report in Pull Request

   This is the report that shows training graph ans testing results. 
   
   * For example this result is not as good as the other one that we have. Because this model training has an %80 accuracy and from the graph it can be said that if it is a good one, training and validation lines should be almost paralel to each other. At least they should increase together and decrease together. So pull request can be rejected.

     ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss23.png)

     ![Image](https://github.com/tuvisai/case-study-cicd-ml-with-jenkins/blob/49b1edf0492a3d0c7e026f741c59002b59434318/images/ss24.png)




<a name="Conclusion-Suggestions"></a>
## Conclusion-Suggestions

   This can be used for different `feature branches`. Assume, there is a different ideas about the model and each developer works in a different branch. Each time they made a `pull request`, the result about the model will appear. And based on those results, best model can be choose as the project model and merge with `develop` branch and finally can be the last version of the project in `main` branch. 



<a name="## Sources"></a>
## Sources

https://www.tensorflow.org/tutorials/images/classification
https://www.tensorflow.org/tutorials/keras/classification
https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/16_cnn_cifar10_small_image_classification/cnn_cifar10_dataset.ipynb
https://xaviervasques.medium.com/quick-install-and-first-use-of-docker-327e88ef88c7
https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f
https://aws.amazon.com/blogs/opensource/why-use-docker-containers-for-machine-learning-development/
https://towardsdatascience.com/have-your-ml-models-built-automatically-using-github-actions-5caa03c6f571
https://towardsdatascience.com/from-devops-to-mlops-integrate-machine-learning-models-using-jenkins-and-docker-79034dbedf1 
https://www.youtube.com/playlist?list=PL7WG7YrwYcnDBDuCkFbcyjnZQrdskFsBz 






