FROM ubuntu:22.04
LABEL authors="seominjae/dockerfile"

ENTRYPOINT ["top", "-b"]
