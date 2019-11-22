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
                    sh 'make all docs -j8 VERBOSE=1' // CXXFLAGS="-fdiagnostics-color=always" 
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
            steps {
                sh 'rm -rf build'
                dir("build") {
                    sh 'cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DBUILD_ALL_EXAMPLES=OFF -BUILD_DOCUMENTATION=OFF -DSMS="52"'
                    sh 'make tests -j8'
                    sh 'valgrind --suppressions=../tools/valgrind-cuda-suppression.supp --error-exitcode=1 --leak-check=full --gen-suppressions=no ./bin/linux-x64/Debug/tests'
                }
            }
        }
    }
}