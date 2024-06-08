pipeline {
    agent any

    stages {
        stage('Git') {
            steps {
                git branch: 'main', url: 'https://github.com/timurgainutdinov1/birds_counter_project.git'
            }
        }
        stage('Streamlit run') {
            steps {
                sh '''streamlit run st_birds_counter.py'''
            }
        }

    }
}