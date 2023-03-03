\ Model RRP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 x_kimjn[0,1,0,1,0] + 0 x_kimjn[0,2,0,1,0] + 0 x_kimjn[0,2,0,2,0]
   + z_ki[0,1]
Subject To
 R0: xs_kim[0,1,0] + xod_k[0] = 1
 R1: xe_kjn[0,2,0] + xod_k[0] = 1
 (4.5)[0,1,0]: - x_kimjn[0,1,0,2,0] + xs_kim[0,1,0] = 0
 (4.6)[0,2,0]: - x_kimjn[0,1,0,2,0] + xe_kjn[0,2,0] = 0
 (4.7)[0,1,0]: xs_kim[0,1,0] - yp_kim[0,1,0] = 0
 (4.8)[2,0,0]: xe_kjn[0,2,0] - yd_kim[0,2,0] = 0
 R6: yp_kim[0,1,0] <= 1
 R7: yd_kim[0,2,0] <= 1
 R8: xs_kim[0,1,0] - z_ki[0,1] = 0
 R9: xod_k[0] = 0
 R10: 2 x_kimjn[0,1,0,2,0] = 0
 R11: t_ki[0,1,0] - t_ki[0,2,0] <= -18
 R12: 375 x_kimjn[0,1,0,2,0] + t_ki[0,1,0] - t_ki[0,2,0] <= 357
 R13: - 375 x_kimjn[0,1,0,2,0] + t_ki[0,1,0] - t_ki[0,2,0] >= -393
 R14: 375 xs_kim[0,1,0] + t_ki[0,0,0] - t_ki[0,1,0] <= 368
 R15: - 375 xs_kim[0,1,0] + t_ki[0,0,0] - t_ki[0,1,0] >= -382
 R16: 375 xe_kjn[0,2,0] + t_ki[0,2,0] - t_ki[0,3,0] <= 374.9
 R17: - 375 xe_kjn[0,2,0] + t_ki[0,2,0] - t_ki[0,3,0] >= -375.1
 R18: t_ki[0,2,0] >= 119.9
 R19: t_ki[0,2,0] <= 149.9
 R20: t_ki[0,3,0] >= 120
 R21: t_ki[0,3,0] <= 150
 R22: - t_ki[0,1,0] + t_ki[0,2,0] <= 126.6666666666667
 R23: - t_ki[0,0,0] + t_ki[0,3,0] <= 150
 R24: 0.1 yp_kim[0,1,0] <= 0
 R25: - 0.1 yd_kim[0,2,0] >= 0
 R26: z_ki[0,1] <= 4
Bounds
Binaries
 x_kimjn[0,1,0,1,0] x_kimjn[0,1,0,2,0] x_kimjn[0,2,0,1,0]
 x_kimjn[0,2,0,2,0] xs_kim[0,1,0] xe_kjn[0,2,0] xod_k[0] yp_kim[0,1,0]
 yd_kim[0,2,0] z_ki[0,1]
End
