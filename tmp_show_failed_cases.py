# -*- coding: utf-8 -*-  
import json, urllib.request  
API='http://127.0.0.1:8000/v1/tasks/run'  
q1='\u8bf7\u6d4b\u8bd5\u8be5\u8fd0\u653e\u7535\u8def\u7684\u5e26\u5bbd'  
q2='\u73af\u8def\u589e\u76ca\u6d4b\u8bd5\u5e94\u8be5\u600e\u4e48\u505a'  
cases=[('opamp_bandwidth_cn',q1),('loop_gain_cn',q2)]  
for cid,q in cases:  
    payload={'query':q,'circuit_description':'\u4e24\u7ea7\u8fd0\u653e\uff0c\u8f93\u51fa\u5e26\u8d1f\u8f7d','top_k':6,'execute':False}  
    req=urllib.request.Request(API,data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')  
    with urllib.request.urlopen(req,timeout=180) as r:  
        data=json.loads(r.read().decode('utf-8'))  
    print('\n=== %s ===' % cid)  
    print('stderr=',(data.get('logs',{}) or {}).get('stderr',''))  
    ev=data.get('evidence',[])  
    for i,e in enumerate(ev,1):  
        src=e.get('source','')  
        sn=(e.get('snippet','') or '').replace('\n',' ')[:220]  
        print(str(i)+'. '+src)  
        print('   '+sn)  
