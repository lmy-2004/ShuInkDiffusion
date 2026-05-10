[Transition] 在这个定位下，为什么我们选择 Live2Diff 而不是别的基线？

结论很直接，因为它已经解决了实时视频扩散里最难的共性问题。比如单向时序注意力、warmup、KV-cache、Depth Prior，以及对 DreamBooth 和 LoRA 的兼容，再加上 TinyVAE 和 TensorRT 这样的加速路径，使它成为一个能真正跑进实时区间的底座。 [Data] 换句话说，Live2Diff 的价值不在于它已经会画水墨，而在于它已经把“实时视频扩散能跑起来”这件事做成立了。

Key points: ① Live2Diff 已具备关键实时机制 ② 它定义了项目的技术边界 ③ 它提供的是底座而不是现成水墨方案
Duration: 1.2 minutes