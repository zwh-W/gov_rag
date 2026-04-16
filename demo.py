from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
# 强制删除有问题的旧索引
es.indices.delete(index="chunk_info_v2", ignore_unavailable=True)
es.indices.delete(index="document_meta_v2", ignore_unavailable=True)
print("✅ 旧索引已被彻底清除！")