\ Model RRP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 x_kimjn[0,1,0,1,0] + 0 x_kimjn[0,2,0,1,0] + 0 x_kimjn[0,2,0,2,0]
   + z_ki[0,1] + 0 t_ki[0,0,0] + 0 t_ki[0,1,0] + 0 t_ki[0,2,0]
   + 0 t_ki[0,3,0]
Subject To
 ny1: xs_kim[0,1,0] = 1
 ny2: x_kimjn[0,1,0,2,0] = 1
 Ny3: xe_kjn[0,2,0] = 1
 4.3[0]: xs_kim[0,1,0] + xod_k[0] = 1
 4.4[0]: xe_kjn[0,2,0] + xod_k[0] = 1
 (4.7)[0,1,0]: xs_kim[0,1,0] - yp_kim[0,1,0] = 0
 (4.8)[2,0,0]: xe_kjn[0,2,0] - yd_kim[0,2,0] = 0
 R7: yp_kim[0,1,0] - yp_kim[0,2,0] = 0
 4.12[0,1]: xod_k[0] + z_ki[0,1] <= 1
 4.11[0,1]: - yp_kim[0,1,0] + z_ki[0,1] = 0
 4.13[0,1]: 2 x_kimjn[0,1,0,2,0] = 0
 R11: z_ki[0,1] <= 4
Bounds
Binaries
 x_kimjn[0,1,0,1,0] x_kimjn[0,1,0,2,0] x_kimjn[0,2,0,1,0]
 x_kimjn[0,2,0,2,0] xs_kim[0,1,0] xe_kjn[0,2,0] xod_k[0] yp_kim[0,1,0]
 yp_kim[0,2,0] yp_kim[0,1,0] yd_kim[0,2,0] z_ki[0,1]
End
