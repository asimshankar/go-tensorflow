set -e

go get github.com/golang/protobuf/proto
go get github.com/golang/protobuf/protoc-gen-go

cd $(dirname $0)
TF_DIR=${GOPATH}/src/github.com/tensorflow/tensorflow

PROTOC=$(which protoc)
if [ ! -x "${PROTOC}" ]
then
  echo "Protocol buffer compiler protoc not found in PATH"
  echo "Perhaps install it from https://github.com/google/protobuf/releases"
  exit 1
fi

# Ensure that protoc-gen-go is available in $PATH
# Since ${PROTOC} will require it.
export PATH=$PATH:${GOPATH}/bin
mkdir -p ./proto
${PROTOC} \
  -I ${TF_DIR} \
  --go_out=./proto \
  ${TF_DIR}/tensorflow/core/example/{example,feature}.proto
