## 编写节点nod.xml,车道edg.xml,连接con.xml，灯组tll.xml


## 生成网络模型net.xml

```
netconvert --node-files=zfdx.nod.xml \
        --edge-files=zfdx.edg.xml \
        --connection-files=zfdx.con.xml \
        --tllogic-files=zfdx.tll.xml \
        --output-file=zfdx.net.xml \
        --ignore-errors
```

## 编写需求模型rou.xml

## 编写zfdx.sumocfg

```
<configuration>
    <input>
        <net-file value="zfdx.net.xml"/>
        <route-files value="zfdx-perhour.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="100000"/>
    </time>
    <viewsettings>
        <scheme name="real world"/>
    </viewsettings>
</configuration>
```