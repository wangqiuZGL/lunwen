# 2026-11-18 服务器同步记录

只读拉取服务器 `/root/lunwen`；训练期间未修改远端代码、配置或进程。

本机完整快照：`C:/Users/LH/Desktop/lunwen/server-sync/20261118/source`。
原始控制台/环境/代码ZIP等：同级 `training-records`。

292份源码文件通过服务器生成的逐文件SHA256校验。忽略LF/CRLF后，服务器与整合仓库的内容差异仅8处，均已核对：本机配对功能、README/说明、torchvision版本判断修复及两份启动脚本的引号差异。服务器两份启动脚本按原文同步；其他差异保留本机新功能。不存在尚未合入的服务器训练修复。Notebook自动保存的README检查点只保留在完整快照。

详细来源和哈希：[server-sync-manifest.json](server-sync-manifest.json)。

原始阶段日志：[training-logs](training-logs)。日志字节不经Git换行转换。配置Namespace与逐轮JSON原文保留，融合微调日志包含batch8/累积4到batch16/累积2的恢复记录。

| 实验 | 已完成微调轮数 | 最佳epoch（0起） | Acc@0.5 | 同轮mIoU | 是否最终 |
|---|---:|---:|---:|---:|---|
| 基线 | 8/8 | 7 | 73.3965% | 62.2307% | 是，依据日志 |
| SADB＋TCGS | 8/8 | 7 | 73.2106% | 62.0848% | 是，依据日志 |
| 仅SADB | 6/8 | 5 | 71.3237% | 59.9698% | 否，快照截至北京时间22:58 |

未下载权重，以上不表示重新评估了checkpoint。SADB后续成绩不在本次快照内。CSV保存原始小数精度，不与test结果混用。

推送检查：本机与服务器HTTPS push均因缺少GitHub用户名/认证失败，已连接GitHub账号也无写权限。代码同步已经完成，远端发布仍需要仓库写权限凭据。未修改origin，也未强推。
