\ Model RRP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 x_kimjn[0,1,0,1,0] + 0 x_kimjn[0,2,0,1,0] + 0 x_kimjn[0,2,0,2,0]
   + z_ki[0,1] + 0 t_ki[0,0,0] + 0 t_ki[0,3,0]
Subject To
 R0: xs_kim[0,1,0] + xod_k[0] = 1
 R1: xe_kjn[0,2,0] + xod_k[0] = 1
 R2: - x_kimjn[0,1,0,2,0] + xs_kim[0,1,0] = 0
 (4.6)[0,2,0]: - x_kimjn[0,1,0,2,0] + xe_kjn[0,2,0] = 0
 (4.7)[0,1,0]: xs_kim[0,1,0] - yp_kim[0,1,0] = 0
 (4.8)[2,0,0]: xe_kjn[0,2,0] - yd_kim[0,2,0] = 0
 R6: yp_kim[0,1,0] <= 1
 R7: yd_kim[0,2,0] <= 1
 R8: xs_kim[0,1,0] - z_ki[0,1] = 0
 R9: xod_k[0] + z_ki[0,1] <= 1
 R10: xod_k[0] = 0
 R11: 2 x_kimjn[0,1,0,2,0] = 0
 R12: t_ki[0,1,0] - t_ki[0,2,0] <= -18
 R13: z_ki[0,1] <= 4
Bounds
Binaries
 x_kimjn[0,1,0,1,0] x_kimjn[0,1,0,2,0] x_kimjn[0,2,0,1,0]
 x_kimjn[0,2,0,2,0] xs_kim[0,1,0] xe_kjn[0,2,0] xod_k[0] yp_kim[0,1,0]
 yd_kim[0,2,0] z_ki[0,1]
End
