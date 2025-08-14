# DenoisingGait (CVPR'25)
### For `00-DenoisingFea_96x48.pkl`

For the data, we preserved the diffusion features and fed them into the subsequent recognition network.  
In our initial configuration, we adopted **Stable Diffusion v1.5**, set the inference loop to **1**, and fixed \( t = 700 \) to extract the denoised features.  
The input resolution was \((3, 768, 384)\) \((C, H, W)\), and the resulting feature map had the shape \((4, 96, 48)\).  
These features, together with the silhouette inputs, were then passed to the downstream **DenoisingGait** network.



```
data_for_DenoisingGait
├── 0000                            # Identity
│   ├── 00-nm                       # sequence_number - sequence_covariates
│   │   ├── 000                     # viewpoint_angle
│   │   │   ├── 00-DenoisingFea_96x48.pkl                
│   │   │   └── 01-Sils_96x48.pkl       
                ......
            ......
        ......
    ......
```

