NAME RRP
OBJSENSE MAX
ROWS
 N  OBJ
 E  R0      
 E  R1      
 E  R2      
 E  (4.6)[0,2,0]
 E  (4.7)[0,1,0]
 E  (4.8)[2,0,0]
 L  R6      
 L  R7      
 E  R8      
 L  R9      
 E  R10     
 E  R11     
 L  R12     
 L  R13     
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    x_kimjn[0,1,0,1,0]  OBJ       0
    x_kimjn[0,1,0,2,0]  R2        -1
    x_kimjn[0,1,0,2,0]  (4.6)[0,2,0]  -1
    x_kimjn[0,1,0,2,0]  R11       2
    x_kimjn[0,2,0,1,0]  OBJ       0
    x_kimjn[0,2,0,2,0]  OBJ       0
    xs_kim[0,1,0]  R0        1
    xs_kim[0,1,0]  R2        1
    xs_kim[0,1,0]  (4.7)[0,1,0]  1
    xs_kim[0,1,0]  R8        1
    xe_kjn[0,2,0]  R1        1
    xe_kjn[0,2,0]  (4.6)[0,2,0]  1
    xe_kjn[0,2,0]  (4.8)[2,0,0]  1
    xod_k[0]  R0        1
    xod_k[0]  R1        1
    xod_k[0]  R9        1
    xod_k[0]  R10       1
    yp_kim[0,1,0]  (4.7)[0,1,0]  -1
    yp_kim[0,1,0]  R6        1
    yd_kim[0,2,0]  (4.8)[2,0,0]  -1
    yd_kim[0,2,0]  R7        1
    z_ki[0,1]  OBJ       1
    z_ki[0,1]  R8        -1
    z_ki[0,1]  R9        1
    z_ki[0,1]  R13       1
    MARKER    'MARKER'                 'INTEND'
    t_ki[0,0,0]  OBJ       0
    t_ki[0,1,0]  R12       1
    t_ki[0,2,0]  R12       -1
    t_ki[0,3,0]  OBJ       0
RHS
    RHS1      R0        1
    RHS1      R1        1
    RHS1      R6        1
    RHS1      R7        1
    RHS1      R9        1
    RHS1      R12       -18
    RHS1      R13       4
BOUNDS
 BV BND1      x_kimjn[0,1,0,1,0]
 BV BND1      x_kimjn[0,1,0,2,0]
 BV BND1      x_kimjn[0,2,0,1,0]
 BV BND1      x_kimjn[0,2,0,2,0]
 BV BND1      xs_kim[0,1,0]
 BV BND1      xe_kjn[0,2,0]
 BV BND1      xod_k[0]
 BV BND1      yp_kim[0,1,0]
 BV BND1      yd_kim[0,2,0]
 BV BND1      z_ki[0,1]
ENDATA
