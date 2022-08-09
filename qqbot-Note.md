

# qq机器人
依赖项目：
- [FloatTech/ZeroBot-Plugin](https://github.com/FloatTech/ZeroBot-Plugin)
- [Mrs4s/go-cqhttp](https://github.com/Mrs4s/go-cqhttp)
  - [go-cqhttp/releases](https://github.com/Mrs4s/go-cqhttp/releases) 包下载页面
- [jagaimotomato/cqhttp_reboot](https://github.com/jagaimotomato/cqhttp_reboot)

部署步骤：
- 部署go-cqhttp
   - 下载go-cqhttp,根据操作系统选择，比如mac air m1 选 [go-cqhttp_darwin_arm64.tar.gz](https://github.com/Mrs4s/go-cqhttp/releases/download/v1.0.0-rc3/go-cqhttp_darwin_arm64.tar.gz)
   - 解压文件到合适的目录  tar -xzvf 
   - 启动服务。参考[使用文档](https://docs.go-cqhttp.org/guide/quick_start.html#%E4%BD%BF%E7%94%A8)   
     - 启动服务  ./go-cqhttp
     - 选择2,正向Websocket服务，然后会生成一个config.yaml配置文件
     - 修改config.yaml配置文件，比如账号密码
     - 再次运行 ./go-cqhttp
- 部署ZeroBot-Plugin
  - 下载代码  [FloatTech/ZeroBot-Plugin](https://github.com/FloatTech/ZeroBot-Plugin)
  - 拉取子模块 git submodule update --init --recursive
    - 下载慢的话可以直接下载发布包wget https://github.com/FloatTech/zbpdata/archive/refs/tags/v0.0.28.tar.gz
    - 解压到data目录
  - 运行脚本试试 sh run.sh 
  - 创建一个配置文件config.json，样例在readme里面有
  - 然后运行 go run main.go config.go -c config.json
  - 在linux机器上运行
    - 编译 GOOS=linux GOARCH=amd64 go build -ldflags "-s -w" -o zerobot -trimpath
    - 运行 nohup ./zerobot  -c ./config.json >> /data/potter/logs/zerobot.log 2>&1&
- 部署cqhttp_reboot