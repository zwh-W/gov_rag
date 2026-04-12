# 1. 基于官方的 Elasticsearch 8.5.3 镜像
FROM docker.elastic.co/elasticsearch/elasticsearch:8.5.3

# 🚀 新增：临时切换到最高权限(root)管理员账户
USER root

# 2. 将我们下载的插件压缩包，复制到容器的临时目录中
COPY ./elasticsearch-analysis-ik-8.5.3.zip /tmp/ik.zip

# 3. 运行命令，在容器内解压插件到正确的 plugins 目录
RUN \
  cd /tmp && \
  unzip ik.zip -d /usr/share/elasticsearch/plugins/ik && \
  rm ik.zip && \
  chown -R elasticsearch:elasticsearch /usr/share/elasticsearch/plugins/ik

# 🚀 新增：操作完成后，切回普通的 elasticsearch 用户，保证系统安全
USER elasticsearch