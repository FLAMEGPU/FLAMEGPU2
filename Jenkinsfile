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
        
        stage('Lint') {
            steps {
                dir("build") {
                    sh 'make all_lint' 
                }
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
                    sh 'make all docs -j8' // CXXFLAGS="-fdiagnostics-color=always" 
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
        
        stage('MemCheck') {
                dir("build") {
                    sh 'cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON'
                    valgrind --suppressions=../tools/valgrind-cuda-suppression.supp --error-exitcode=1 --leak-check=full --gen-suppressions=no ./bin/linux-x64/Debug/tests
                }        
        }
    }
}