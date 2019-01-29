# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#!/bin/bash
CURR_DIR=$(cd $(dirname $0)/../; pwd)
CLASSPATH=$CLASSPATH:$CURR_DIR/target/*:$CLASSPATH:$CURR_DIR/target/dependency/*
java -Xmx8G -cp $CLASSPATH mxnet.EndToEndModelWoPreprocessing --num-runs 100 --batchsize 32 --model-path-prefix ~/incubator-mxnet/scala-package/mxnet-demo/java-demo/models/gluoncv-resnet-18/resnet18_v1 --input-image ~/incubator-mxnet/scala-package/mxnet-demo/java-demo/images/kitten.jpg
