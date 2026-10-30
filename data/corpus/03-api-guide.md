# 云雀协作 开放 API 接入指南

## 认证

所有接口使用 Bearer Token 认证，Token 在「空间设置 → 开放平台 → 访问令牌」创建。请求头格式：

```
Authorization: Bearer yq_live_xxxxxxxxxxxx
```

Token 默认有效期 365 天，可设置只读或读写范围。Token 泄露后应立即吊销，吊销后 30 秒内全网生效。

## 速率限制

- 免费版：每分钟 60 次请求
- 团队版：每分钟 600 次请求
- 专业版：每分钟 3000 次请求
- 企业版：按合同约定，默认每分钟 6000 次请求

超限返回 HTTP 429，响应头 `Retry-After` 给出建议等待秒数。建议指数退避重试，初始 1 秒，最多重试 5 次。

## 常用接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | /open/v1/tasks | 分页查询任务，支持 updated_since 增量同步 |
| POST | /open/v1/tasks | 创建任务 |
| PATCH | /open/v1/tasks/{id} | 更新任务字段 |
| GET | /open/v1/sprints | 查询迭代 |
| POST | /open/v1/webhooks | 注册事件回调 |

分页采用游标分页，单页最多 200 条，游标通过响应体 `next_cursor` 返回。

## Webhook 事件

支持 `task.created`、`task.updated`、`task.completed`、`comment.created`、`sprint.started`、`sprint.closed` 六类事件。回调超时时间为 5 秒，失败后按 1 分钟、5 分钟、30 分钟重试三次，仍失败则标记为死信并可在控制台重放。签名使用 HMAC-SHA256，密钥为 Webhook Secret，签名放在 `X-Yunque-Signature` 头。

## 错误码

- 400 参数错误
- 401 认证失败
- 403 权限不足
- 404 资源不存在
- 409 并发冲突，建议拉取最新版本后重试
- 429 触发限流
- 500 服务内部错误，可携带 request_id 联系支持
