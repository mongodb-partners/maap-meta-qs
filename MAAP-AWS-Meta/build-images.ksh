#!/usr/bin/env bash

cd main
docker build  --tag 'main' .

cd ../ui 
docker build  --tag 'ui' .

cd ../loader 
docker build  --tag 'loader' .

cd ../logger 
docker build  --tag 'logger' .

cd ../ai-memory 
docker build  --tag 'ai-memory' .

cd ../semantic-cache
docker build  --tag 'semantic-cache' .

cd ../nginx 
docker build  --tag 'nginx' .