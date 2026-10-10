# 平台(v7_single 测量机器,§3 补测第 1 项)

采集时间 2026-10-10T15:46:51,提交 v7-rc1-18-g5174642(`5174642b3c8c`)。

**一句话**:AMD Ryzen 9 9950X 16-Core Processor(16 核 32 线程,WMI 最大时钟 4300 MHz,L3 64 MB),31.1 GiB 内存(32 GB F5-6400J3239G32G (DIMM 1, 4800 MT/s)),ASUSTeK COMPUTER INC. ProArt X870E-CREATOR WIFI,Microsoft Windows 11 Pro 25H2(build 26200.7171);两张 NVIDIA GeForce RTX 5090(各 32607 MiB,功耗上限 600.00 W,PCIe Gen 5 x16 槽位、当前链路宽度 x8),测量用的是无头的那张(PCI 00000000:03:00.0,uuid ae137c66…;显示器接在 PCI 00000000:01:00.0 的那张);驱动 576.88(Vulkan driverVersion 576.88.0.0,apiVersion 1.4.303,conformance 1.4.1.0),Vulkan loader 1.4.350.0(SDK 1.4.350.0,vulkaninfo 实例版本 1.4.350);Python 3.13.9,python-vulkan 1.3.275.1,numpy 2.4.6,scipy 1.17.1,glslc shaderc v2026.2 v2026.2。

| 项 | 值 |
|---|---|
| CPU | AMD Ryzen 9 9950X 16-Core Processor:16 核 / 32 线程,WMI MaxClockSpeed 4300 MHz,L2 16 MB,L3 64 MB |
| 内存 | 31.1 GiB 可用;32 GB F5-6400J3239G32G (DIMM 1, 4800 MT/s) |
| 主板 | ASUSTeK COMPUTER INC. ProArt X870E-CREATOR WIFI |
| Windows | Microsoft Windows 11 Pro 25H2,版本 10.0.26200,build 26200.7171,64-bit |
| GPU 0(display) | NVIDIA GeForce RTX 5090,PCI 00000000:01:00.0,uuid fb83f0dafd9e1d38dc05e7ae643e22a8,显存 32607 MiB,功耗上限 600.00 W(默认 600.00 W,最大 600.00 W),PCIe Gen max 5(GPU max 5),采集时 Gen 2;宽度 max x16,当前 x8;显示输出 Enabled;最高 SM / 显存时钟 3090 MHz / 14001 MHz;VBIOS 98.02.2e.40.e1 |
| GPU 1(headless (measured)) | NVIDIA GeForce RTX 5090,PCI 00000000:03:00.0,uuid ae137c668a40f5acaab90a83f7cda175,显存 32607 MiB,功耗上限 600.00 W(默认 600.00 W,最大 600.00 W),PCIe Gen max 5(GPU max 5),采集时 Gen 1;宽度 max x16,当前 x8;显示输出 Disabled;最高 SM / 显存时钟 3090 MHz / 14001 MHz;VBIOS 98.02.2e.40.e1 |
| 驱动 | 576.88(CUDA 12.9) |
| vulkaninfo GPU0 | NVIDIA GeForce RTX 5090:apiVersion 1.4.303,driverVersion 576.88.0.0,driverInfo 576.88,conformance 1.4.1.0,uuid fb83f0da-fd9e-1d38-dc05-e7ae643e22a8 |
| vulkaninfo GPU1 | AMD Radeon(TM) Graphics:apiVersion 1.4.315,driverVersion 2.0.353,driverInfo 26.6.1 (AMD proprietary shader compiler),conformance 1.4.0.0,uuid 00000000-7a00-0000-0000-000000000000 |
| vulkaninfo GPU2 | NVIDIA GeForce RTX 5090:apiVersion 1.4.303,driverVersion 576.88.0.0,driverInfo 576.88,conformance 1.4.1.0,uuid ae137c66-8a40-f5ac-aab9-0a83f7cda175 |
| Vulkan loader / SDK | vulkan-1.dll 1.4.350.0;VULKAN_SDK C:\VulkanSDK\1.4.350.0;vulkaninfo 实例版本 1.4.350 |
| glslc | shaderc v2026.2 v2026.2; spirv-tools v2026.2 v2022.4-1193-gc1cb30bb; glslang 11.1.0-1412-g5ed4003a; Target: SPIR-V 1.0 |
| Python | 3.13.9(C:\Users\cchen\PycharmProjects\vulkan-demo\.venv\Scripts\python.exe);vulkan 1.3.275.1, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.0, PyYAML 6.0.3 |
| 提交 | v7-single v7-rc1-18-g5174642 5174642b3c8c22ff778df51c43823be9309ceb3e |
