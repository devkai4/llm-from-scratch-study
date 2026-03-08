import requests

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)

response = requests.get(url, timeout=30)

print(type(response))
print(response.status_code)
print(response.headers)
print(response.text[:200])

# ---
# <class 'requests.models.Response'>
# 200
# {'Connection': 'keep-alive', 'Content-Length': '8891', 'Cache-Control': 'max-age=300', 'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'; sandbox", 'Content-Type': 'text/plain; charset=utf-8', 'ETag': 'W/"8fd4f804f4b2fdd81bb9e93a4783169aba85fbaa1d198e1c6e52839895c8db02"', 'Strict-Transport-Security': 'max-age=31536000', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'deny', 'X-XSS-Protection': '1; mode=block', 'X-GitHub-Request-Id': 'AA62:C41C9:23465F:5452DF:69A3E78D', 'Content-Encoding': 'gzip', 'Accept-Ranges': 'bytes', 'Date': 'Sun, 01 Mar 2026 07:15:26 GMT', 'Via': '1.1 varnish', 'X-Served-By': 'cache-nrt-rjtt7900072-NRT', 'X-Cache': 'MISS', 'X-Cache-Hits': '0', 'X-Timer': 'S1772349326.960577,VS0,VE183', 'Vary': 'Authorization,Accept-Encoding', 'Access-Control-Allow-Origin': '*', 'Cross-Origin-Resource-Policy': 'cross-origin', 'X-Fastly-Request-ID': '0959b009f06d5e88168f03530c2af66b1d70a130', 'Expires': 'Sun, 01 Mar 2026 07:20:26 GMT', 'Source-Age': '0'}
# I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a

class Response:
    # データ
    self.status_code = ...
    self.content = ...
    self.text = ...
    self.headers = ...
    
    # 機能
    def raise_for_status(self):
        ...