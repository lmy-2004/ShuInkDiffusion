[Transition] 在这条实时链路里，真正的中枢不是 wrapper，也不是 demo，而是 Streaming UNet。

原因在于，当前帧的 noisy latent、depth latent、文本条件、历史 KV-cache、时间窗口规则以及缓存写回位置，全部都在这里汇合。换句话说，这里的 UNet 已经不再只是一个“图像去噪网络”，而是一个以 latent 为主干、以时序注意力为记忆机制、以 depth 为结构偏置的流式视频去噪器。这个判断对后面的改造方向非常关键。 [Pause]

Key points: ① 所有关键状态都汇合到 UNet ② 它承担记忆、结构与去噪三重角色 ③ 后续改造应优先落在 UNet 主路径
Duration: 1.3 minutes