pipeline {
    agent { dockerfile true }
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
                    ansiColor('xterm') {
                        sh 'rm -rf build'
                        sh 'mkdir -p build'
                        dir("build") {
                            sh 'cmake .. -DBUILD_TESTS=ON'
                            sh 'make all docs -j8' 
                            archiveArtifacts artifacts: '**/bin/linux-x64/Release/*', fingerprint: true
                        }
                    }
                }
            }

            stage('Test') {
                steps {
                    ansiColor('xterm') {
                        sh 'ls build/bin/linux-x64/Release/'
                        sh './build/bin/linux-x64/Release/tests'
                    }
                }
            }
            
            stage('Lint') {
                steps {
                    ansiColor('xterm') {
                        sh 'make all_lint' 
                    }
                }
            }
            post {
                always {
                    stage('Cleanup') {
                        step([$class: 'WsCleanup'])
                    }
                }
            }
        }
    }
}
