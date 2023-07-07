
# Adappt - Machine Learning Tasks

To predict the attrition risks (exited) for the employees 




## Authors

- [@Sudharmendra1](https://github.com/Sudharmendra1)


# Installation

Install necessary packages for Machine Learning:
Latest Versions are recommended

1. Install VSCode/Anaconda - Jupyter Notebook: https://code.visualstudio.com/download , https://www.anaconda.com/download 
2. Numpy: pip install numpy 
3. Pandas: pip install pandas 
4. Matplotlib: pip install matplotlib
5. Scikit: pip install -U scikit-learn
6. Seaborn: pip install seaborn
7. Docker Installation: https://docs.docker.com/engine/install/

## Deployment

1. To run the ML model: Import all the necessary packages as mentioned in the Installation section.

Use: python3 employee.py

2. To run the Docker through CLI: 
a. Open Terminal. Change the directory where it contains the model and datasets. 
b. Create a docker file, by entering the below commands. Copy paste it and modify the directory

# Use a base image with the desired operating system and runtime
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /Users/sudharmendragv/Downloads/Adappt

# Copy the ML model files to the working directory
#COPY /Users/sudharmendragv/Downloads/Adappt .
COPY requirements.txt .


# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose any necessary ports (if applicable)
#EXPOSE 8000

# Define the command to run your ML model
CMD ["python", "employee.py"]

c. Save the dockerfile in the same directory and create a requirement file with all the necessary packages details. This should look like below.
requirements.txt

numpy==1.19.5
pandas==1.3.4
scikit-learn==1.0
matplotlib==3.4.3
seaborn==0.11.2
scipy==1.7.1

d. Save the requirements.txt file and run the below command.
Build the docker image: "docker build -t ml_model ."
Note: "ml_model" - Name of the docker image. Any name can be provided
Run the docker image: docker run ml_model




## Run Locally

Clone the project

Link: git clone https://github.com/Sudharmendra1/Adappt-Tasks.git

Follow the Installment and Deployment section

## Optimizations

1. Implemented couple of models: Logistic Regression, Linear SVM, Decision Tree and Random Forest. Obtained an accuracy of 87%

2. Inorder to obtain the same accuracy, hypertuning of the parameters were done like:

    a. For Logistic Regression: Tuning the RandomState value
    b. For Linear SVM: Tuning the learning rate and             regularization parameters




## Screenshots

Output Samples in CLI: "https://github.com/Sudharmendra1/Adappt-Tasks/blob/main/CLI%20-%20Output.png"

Output Samples in Docker: "https://github.com/Sudharmendra1/Adappt-Tasks/blob/main/Docker%20-%20Output.png"

Docker Image: "https://github.com/Sudharmendra1/Adappt-Tasks/blob/main/Docker%20-%20Image.png"




## Acknowledgements

 - [Machine Learning model deployment in Docker](https://www.analyticsvidhya.com/blog/2022/05/a-complete-guide-for-deploying-ml-models-in-docker/)
 - [Machine Learning model deployment using Flask and Docker](https://www.turing.com/kb/deploy-ml-models-with-flask-and-docker)

## Support

Any issues: Please write an email to this email address:  gvsudharmendra31@gmail.com

