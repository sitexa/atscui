# 郑州新港大道与遵大路交叉口 路网模型与需求模型 设计

### 1， 定义节点 xgzd.nod.xml

``` 
<nodes>
    <node id="n" x="0.0" y="300.0" type="priority"/>
    <node id="s" x="0.0" y="-300.0" type="priority"/>
    <node id="e" x="300.0" y="0.0" type="priority"/>
    <node id="w" x="-300.0" y="0.0" type="priority"/>
    <node id="t" x="0.0" y="0.0" type="traffic_light" tlType="actuated" tl="tl_1"/>
</nodes>
```


### 2， 定义路连接 xgzd.edg.xml

```
<edges>
    <edge id="n_t" from="n" to="t" numLanes="3"/>
    <edge id="t_n" from="t" to="n" numLanes="3"/>
    <edge id="w_t" from="w" to="t" numLanes="4"/>
    <edge id="t_w" from="t" to="w" numLanes="3"/>
    <edge id="s_t" from="s" to="t" numLanes="3"/>
    <edge id="t_s" from="t" to="s" numLanes="3"/>
    <edge id="e_t" from="e" to="t" numLanes="4"/>
    <edge id="t_e" from="t" to="e" numLanes="3"/>
</edges>
```

### 3， 定义车道连接 xgzd.con.xml *车道编号：从外向内:0,1,2...*

车道数据： 北3，东4，南3，西4

```
<connections>
    <!--从北到南,右转,无灯控-->
    <!--从北到南,直行-->
    <connection from="n_t" to="t_s" fromLane="0" toLane="0"/>
    <connection from="n_t" to="t_s" fromLane="1" toLane="1"/>
    <!--从北到南,左转-->
    <connection from="n_t" to="t_e" fromLane="2" toLane="2"/>
    <!--从北到南,调头,无-->

    <!--从南到北，右转,无灯控-->
    <!--从南到北，直行-->
    <connection from="s_t" to="t_n" fromLane="0" toLane="0"/>
    <connection from="s_t" to="t_n" fromLane="1" toLane="1"/>
    <!--从南到北，左转-->
    <connection from="s_t" to="t_w" fromLane="2" toLane="2"/>
    <!--从南到北，调头,无-->

    <!--从东到西，右转，无灯控-->
    <!--从东到西，直行-->
    <connection from="e_t" to="t_w" fromLane="0" toLane="0"/>
    <connection from="e_t" to="t_w" fromLane="1" toLane="1"/>
    <!--从东到西，左转-->
    <connection from="e_t" to="t_s" fromLane="2" toLane="2"/>
    <connection from="e_t" to="t_s" fromLane="3" toLane="2"/>
    <!--从东到西，调头，无-->

    <!--从西到东，右转，无灯控-->
    <!--从西到东，直行-->
    <connection from="w_t" to="t_e" fromLane="0" toLane="0"/>
    <connection from="w_t" to="t_e" fromLane="1" toLane="1"/>

    <!--从西到东，左转-->
    <connection from="w_t" to="t_n" fromLane="2" toLane="2"/>
    <connection from="w_t" to="t_n" fromLane="3" toLane="2"/>
    <!--从西到东，调头，无-->

</connections>
```

### 4， 定义灯组信息 xgzd.tll.xml

```
<tlLogic id="tl_1" programID="0" offset="0" type="static">
    <phase duration="43" state="GGrrrrrGGrrrrr"/>
    <phase duration="3"  state="yyyrrrryyyrrrr"/>
    <phase duration="16" state="rrGrrrrrrGrrrr"/>
    <phase duration="3"  state="rryrrrrrryrrrr"/>
    <phase duration="53" state="rrrGGrrrrrGGrr"/>
    <phase duration="3"  state="rrryyrrrrryyrr"/>
    <phase duration="16" state="rrrrrGGrrrrrGG"/>
    <phase duration="3"  state="rrrrryyrrrrryy"/>
</tlLogic>
```

### 5， 使用netconvert生成路网xgzd.net.xml

```
netconvert --node-files=xgzd.nod.xml \
           --edge-files=xgzd.edg.xml \
           --connection-files=xgzd.con.xml \
           --tllogic-files=xgzd.tll.xml \
           --output-file=xgzd.net.xml \
           --ignore-errors
```

### 6， 编写需求模型 xgzd.rou.xml
``` 
<routes>
    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="30" sigma="0.5" />

    <!--北南直行-->
    <route id="route_ns" edges="n_t t_s"/>
    <!--北南左转-->
    <route id="route_ne" edges="n_t t_e"/>

    <!--东西直行-->
    <route id="route_ew" edges="e_t t_w"/>
    <!--东西左转-->
    <route id="route_es" edges="e_t t_s"/>

    <!--南北直行-->
    <route id="route_sn" edges="s_t t_n"/>
    <!--南北左转-->
    <route id="route_sw" edges="s_t t_w"/>

    <!--西东直行-->
    <route id="route_we" edges="w_t t_e"/>
    <!--西东左转-->
    <route id="route_wn" edges="w_t t_n"/>

    <flow id="flow_ns" route="route_ns" begin="0" end="100000" vehsPerHour="300" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_ne" route="route_ne" begin="0" end="100000" vehsPerHour="50" departSpeed="max" departPos="base" departLane="best"/>

    <flow id="flow_ew" route="route_ew" begin="0" end="100000" vehsPerHour="500" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_es" route="route_es" begin="0" end="100000" vehsPerHour="100" departSpeed="max" departPos="base" departLane="best"/>

    <flow id="flow_sn" route="route_sn" begin="0" end="100000" vehsPerHour="300" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_sw" route="route_sw" begin="0" end="100000" vehsPerHour="60" departSpeed="max" departPos="base" departLane="best"/>

    <flow id="flow_we" route="route_we" begin="0" end="100000" vehsPerHour="500" departSpeed="max" departPos="base" departLane="best"/>
    <flow id="flow_wn" route="route_wn" begin="0" end="100000" vehsPerHour="100" departSpeed="max" departPos="base" departLane="best"/>
</routes>
```

### 7， 编写xgzd.sumocfg

``` 
<configuration>
    <input>
        <net-file value="xgzd.net.xml"/>
        <route-files value="xgzd.rou.xml"/>
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

### 7，运行sumo仿真

``` 
sumo-gui xgzd.sumocfg
```