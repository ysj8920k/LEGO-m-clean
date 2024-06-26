def calc_place_offset(x,y):
    
    # Define the points
    point1 = [-34.5,-30]
    point2 = [-24.75,320.5]
    point3 = [316,-40.75]

    stx = 44
    sty = 44
     
    # Define the expected distance between studs
    stud_distance = 8  # in mm

    lx = point3[0]-point1[0]
    ly = point2[1]-point1[1]
    print('ly: ' + str(ly))
    print('lx: ' + str(lx))


    dx = (lx-stud_distance*stx)/(stud_distance*stx)
    dy = (ly-stud_distance*sty)/(stud_distance*sty)
    dxy = (point2[0]-point1[0])/(stud_distance*sty)
    dyx = (point3[1]-point1[1])/(stud_distance*stx)
    print('dx,dy,dxy,dyx:')
    print(str(dx))
    print(str(dy))
    print(str(dxy))
    print(str(dyx))

    x_new = x-0.8 + ((x+point1[0])*(dx+0.002)) + ((y+point1[1])*(dxy-0.0045))
    y_new = y-0.6 + ((y+point1[1])*(dy-0.005)) + ((x+point1[0])*(dyx+0.0045))

    return x_new,y_new

x,y = calc_place_offset(296,96)
print(str(x) + ' and ' + str(y))