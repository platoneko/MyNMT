import urllib.request
import urllib.parse
import json

content = input("请输入需要翻译的内容:\n")
#百度翻译接口
url = "http://fanyi.baidu.com/sug"
#生成一个字典，传输kw键值
data = urllib.parse.urlencode({"kw": content})
headers = {
'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'
}
#生成Request对象
req = urllib.request.Request(url,data=bytes(data,encoding="utf-8"),headers=headers)
r = urllib.request.urlopen(req)
html = r.read().decode('utf-8')
#解析JSON包
html = json.loads(html)
for k in html["data"]:
    print(k["k"],k["v"])
# print(dict_ret)
# print(type(dict_ret))

