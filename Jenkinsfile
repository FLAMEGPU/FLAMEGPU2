pipeline {
    agent { 
        dockerfile true
        // dockerfile {
        //     args '-t'
        // }
    }
    options {
        ansiColor('xterm')
    }
    stages {
        stage('Initialise') {
            steps {
                sh ''
            }
        }

        stage('GPU Check') {
            steps {
                sh 'nvcc --version'
                sh 'nvidia-smi'
            }
        }

        stage('Build') {
            steps {
                sh 'rm -rf build'
                sh 'mkdir -p build'
                dir("build") {
                    sh 'cmake .. -DBUILD_TESTS=ON'
                    sh 'make all docs -j8 // CXXFLAGS="-fdiagnostics-color=always"' 
                    archiveArtifacts artifacts: '**/bin/linux-x64/Release/*', fingerprint: true
                }
            }
        }

        stage('Test') {
            steps {
                sh 'ls build/bin/linux-x64/Release/'
                sh './build/bin/linux-x64/Release/tests'
            }
        }
        
        stage('Lint') {
            steps {
                sh 'make all_lint' 
            }
        }
    }
}