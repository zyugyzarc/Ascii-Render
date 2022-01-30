import os
from libc.stdlib cimport calloc, free
from libc.stdio cimport printf
from libc.math cimport sin, cos
from time import time, sleep

cdef int w = <int>(os.get_terminal_size()[0])
cdef int h = <int>(os.get_terminal_size()[1])

printf("\nsize: %d x %d\n", w, h)

cdef char* screen = <char*> calloc( w*h, sizeof(char) );

cdef void clear():
    cdef int i;
    for i from 0 <= i < w*h by 1:
        screen[i] = b" " 

cdef void show():
    cdef int i;
    for i from 0 <= i < w*h by 1:
        printf("%c", screen[i])
    printf('\n')

cdef void point(int x, int y, char c):
    if 0 < y*w + x and y*w + x < h*w:
        screen[y*w + x]= c

cdef void line(int x1, int y1, int x2, int y2, char ch):
    
    cdef float m, c
    cdef int i =0

    if x1 != x2:

        m = (y2-y1)/(x2-x1)
        c = y1- m*x1
        
        if x1 < x2:
            for i from x1 < i < x2 by 1:
                point(i, <int>(i*m+c), ch)
                

        if x2 < x1:
            for i from x2 < i < x1 by 1:
                point(i, <int>(i*m+c), ch)

    if y1 != y2:

        m = (x2-x1)/(y2-y1)
        c = x1- m*y1
        
        if y1 < y2:
            for i from y1 < i < y2 by 1:
                point(<int>(i*m+c), i, ch)

        if y2 < y1:
            for i from y2 < i < y1 by 1:
                point(<int>(i*m+c), i, ch)

cdef void _flat_bot_tri(int x1, int y1, int x2, int y2, int x3, int y3, char c):

    if y2 != y3: printf("NOT FLAT BOTTOM TRIANGLE"); return
    if y1 == y2: line(x1, y1, x2, y2, c) ; line(x2, y2, x3, y3, c); return

    cdef float m1, c1, m2, c2
    cdef int i, j

    m1 = (x2-x1)/(y2-y1)
    c1 = x1- m1*y1

    m2 = (x3-x1)/(y3-y1)
    c2 = x1- m2*y1


    #printf("p1 <-> p2 : y = (%f)x + %f\n", m1, c1)
    #printf("p1 <-> p3 : y = (%f)x + %f\n", m2, c2)
    #input()

    for i from y1 <= i <= y2 by 1:
        if (i*m1+c1) < (i*m2+c2):
            for j from <int>(i*m1+c1) <= j <= <int>(i*m2+c2) by 1:
                point(j, i, c)
        else:
            for j from <int>(i*m2+c2) <= j <= <int>(i*m1+c1) by 1:
                point(j, i, c)

cdef void _flat_top_tri(int x1, int y1, int x2, int y2, int x3, int y3, char c):

    if y2 != y1: printf("NOT FLAT TOP TRIANGLE"); return
    if y3 == y2: line(x1, y1, x2, y2, c) ; line(x2, y2, x3, y3, c); return

    cdef float m1, c1, m2, c2
    cdef int i, j

    m1 = (x1-x3)/(y1-y3)
    c1 = x3- m1*y3

    m2 = (x3-x2)/(y3-y2)
    c2 = x3- m2*y3

    for i from y1 <= i <= y3 by 1:
        if (i*m1+c1) < (i*m2+c2):
            for j from <int>(i*m1+c1) <= j <= <int>(i*m2+c2) by 1:
                point(j, i, c)
        else:
            for j from <int>(i*m2+c2) <= j <= <int>(i*m1+c1) by 1:
                point(j, i, c)


cdef void triangle(int x1, int y1, int x2, int y2, int x3, int y3, char ch):

    cdef int[3] xi = [x1, x2, x3]
    cdef int[3] yi = [y1, y2, y3]

    cdef float m, c
    cdef int i, j = 0

    for j from 1<= j < 3 by 1:
        for i from 1<= i < 3 by 1:
            #printf("%d ~ %d", yi[i-1], yi[i])
            if yi[i-1] > yi[i]:
                #printf(" <->")
                yi[i-1], yi[i] = yi[i], yi[i-1]
                xi[i-1], xi[i] = xi[i], xi[i-1]
            #printf("\n")

    x1, x2, x3 = xi[0], xi[1], xi[2]
    y1, y2, y3 = yi[0], yi[1], yi[2]
    cdef int x4

    if   y1 == y2: _flat_top_tri(x1, y1, x2, y2, x3, y3, ch)
    elif y2 == y3: _flat_bot_tri(x1, y1, x2, y2, x3, y3, ch)
    
    else:
        m = (x1-x3)/(y1-y3)
        c = x3- m*y3

        x4 = <int>(m*y2 + c)

        _flat_bot_tri(x1, y1, x2, y2, x4, y2, ch)
        _flat_top_tri(x2, y2, x4, y2, x3, y3, ch)

        #point(x4, y2, b'4')

    #point(x1, y1, b'1')
    #point(x2, y2, b'2')
    #point(x3, y3, b'3')


cdef void quad(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, char c):
    triangle(x1, y1, x2, y2, x3, y3, c)
    triangle(x2, y2, x3, y3, x4, y4, c)

cdef float deg(float x): return 180 * x / 3.14159
cdef float rad(float x): return 3.14159 * x / 180


cdef void mat_mul(float* m1, int a1, int b1, float* m2, int a2, int b2, float* m3):
    
    cdef int i, j, k, l
    cdef float s
    # m1 is a1 x b1
    # m2 is a2 x b2

    if b1 != a2: printf("invalid matrix %dx%d x %dx%d\n", a1, b1, a2, b2)

    for i from 0 <= i < a1 by 1:
        for j from 0 <= j < b2 by 1:
            s=0
            for k from 0 <= k < a2 by 1:
                #printf("%f * %f\n", m1[i*b1+k], m2[j+k*b2])
                s = s + m1[i*b1+k] * m2[j+k*b2]
            #printf("= %f\n", s)
            m3[i*b2+j] = s

cdef void rotate_point(float* p, float a, float b, float c):

    cdef float Rx[9]
    cdef float Ry[9]
    cdef float Rz[9]
    cdef int i

    Rx[:] = [1,       0,       0, 
             0,  cos(a), -sin(a),
             0,  sin(a),  cos(a)]

    Ry[:] = [cos(b),  0, -sin(b), 
             0     ,  1,       0,
             sin(b),  0,  cos(b)]

    Rz[:] = [cos(c), -sin(c),  0, 
             sin(c),  cos(c),  0,
             0     ,       0,  1]

    cdef float P1[3]
    cdef float P2[3]

    mat_mul(Rx, 3, 3, p , 3, 1, P1)
    mat_mul(Ry, 3, 3, P1, 3, 1, P2)
    mat_mul(Rz, 3, 3, P2, 3, 1, p )


cdef void project_point(float* p, float sx, float sy):

    cdef float distance = 5.0
    cdef float z = 1/(distance - p[2])
    p[0] = p[0] * z * sx + w/2
    p[1] = p[1] * z * sy + h/2
    #p[2] = p[2]
    p[2] = z


clear()
#triangle(108, 17, 79, 22, 104, 4, b'q')
#_flat_top_tri(0, 0, 5, 0, 5, 5, b'*')
#show()

#input('>')

"""
cdef float points_x[8]
cdef float points_y[8]
cdef float points_z[8]
points_x[:] = [-1, -1, -1, -1,  1,  1,  1, 1]
points_y[:] = [-1,  1, -1,  1, -1, -1,  1, 1]
points_z[:] = [-1, -1,  1,  1, -1,  1, -1, 1]

cdef int edges_a[12]
cdef int edges_b[12]
edges_a[:] = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6]
edges_b[:] = [1, 2, 4, 3, 6, 3, 5, 7, 5, 6, 7, 7]

cdef int faces_a[12]
cdef int faces_b[12]
cdef int faces_c[12]
faces_a[:] = [0, 1, 0, 2, 0, 1, 2, 3, 1, 3, 4, 5]
faces_b[:] = [1, 2, 2, 4, 1, 4, 3, 5, 3, 6, 5, 6]
faces_c[:] = [2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 6, 7]

cdef int normal_x[12]
cdef int normal_y[12]
cdef int normal_z[12]
normal_x[:] = [-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]#[0, 0, 0, 2, 0, 1, 2, 3, 1, 3, 4, 5]
normal_y[:] = [ 0, 0,-1,-1, 0, 0, 0, 0, 1, 1, 0, 0]#[1, 2, 2, 4, 1, 4, 3, 5, 3, 6, 5, 6]
normal_z[:] = [ 0, 0, 0, 0,-1,-1, 1, 1, 0, 0, 0, 0]#[2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 6, 7]

cdef int vcount = 8
cdef int ecount = 12
cdef int fcount = 12

cdef float light[3]
light[:] = [3, 5, -5]

cdef float vbuf_x[8]
cdef float vbuf_y[8]
cdef float vbuf_z[8]
cdef float dbuf_i[12]
cdef float dbuf_d[12]
"""

_points = []
_faces = []
_normals = []
_nmap = []

cdef float points_x[512]
cdef float points_y[512]
cdef float points_z[512]

cdef float normal_x[1024]
cdef float normal_y[1024]
cdef float normal_z[1024]

cdef int faces_a[1024]
cdef int faces_b[1024]
cdef int faces_c[1024]

cdef int nmap[1024]

cdef float vbuf_x[512]
cdef float vbuf_y[512]
cdef float vbuf_z[512]

cdef float dbuf_i[1024]
cdef float dbuf_d[1024]

cdef int vcount = 0
cdef int ecount = 0
cdef int fcount = 0

cdef int _pi, _fi, _ni = 0
with open('object.obj') as f:
    for i_ in f.readlines():
        #printf('%d %d %d\n', _pi, _ni, _fi)

        if i_.startswith('vn'):
            data = i_[3:].strip().replace(' ',',')
            _normals.append( eval(data) )
            normal_x[_ni], normal_y[_ni], normal_z[_ni] = eval(data)
            _ni += 1

        elif i_.startswith('v'):
            data = i_[2:].strip().replace(' ',',')
            _points.append( eval(data) )
            points_x[_pi], points_y[_pi], points_z[_pi] = eval(data)
            _pi += 1


        elif i_.startswith('f'): 
            data = i_[2:].strip().replace(' ',',').replace('//',',')
            data = [j_-1 for j_ in eval(data)]
            _faces.append( data[::2] )
            _nmap.append(data[1])

            nmap[_fi] = data[1]
            faces_a[_fi], faces_b[_fi], faces_c[_fi] = data[0], data[2], data[4]
            _fi += 1

        

    vcount = len(_points)
    fcount = len(_faces)

cdef float light[3]
light[:] = [3, 5, -5]

cdef float lum
cdef int i, j, k = 0
cdef float p[3]
cdef char chars[37]
cdef char c
for i from 0<=i<37 by 1: chars[i] = ".:`\'-,;~_!\"?c\\^<>|=sr1Jo*(C)utia3zLvey75jST{lx}IfY]qp9n0G62Vk8UXhZ4bgdPEKA$wQm&#HDR@WNBM"[i]


printf("loaded model (%d verticies, %d faces)\n", vcount, fcount)#; printf("[")
#for i from 0<= i <fcount by 1 : printf(" %d ", nmap[i])
#printf("]\n")

input('>')
i = 0

try:
    while True:

        t = time()
        clear()

        for j from 0<= j < vcount by 1:

            p[:] = [points_x[j], points_y[j], points_z[j]]
            
            rotate_point (p, rad(90), i/50, 0)
            project_point(p, 100, 100/2)

            vbuf_x[j], vbuf_y[j], vbuf_z[j] = p[:]

        for j from 0<= j < fcount by 1: # fill depth buf

            dbuf_d[j] = ( vbuf_z[faces_a[j]] + vbuf_z[faces_b[j]] + vbuf_z[faces_c[j]] )/3
            #vertex_buf[(faces[j*3])*3+2] + vertex_buf[(faces[j*3]+1)*3+2] + vertex_buf[(faces[j*3]+2)*3+2]
            dbuf_i[j] = j

        # faster
        for j from 0<= j < fcount by 1: # sort depth buf
            for k from 1<= k < fcount by 1:
                if dbuf_d[k-1] > dbuf_d[k]:
                    dbuf_d[k-1], dbuf_d[k] = dbuf_d[k], dbuf_d[k-1]
                    dbuf_i[k-1], dbuf_i[k] = dbuf_i[k], dbuf_i[k-1]

        #printf('[')
        #for j from 0 <= j < fcount by 1:
        #    printf(' %f-%d ', dbuf_d[j], <int>dbuf_i[j])
        #printf(']\n')
        

        #printf("[")
        # render faces
        for j from 0 <= j < fcount by 1:
            
            k = <int>dbuf_i[j]
            lum = ( normal_x[nmap[k]] * light[0] + normal_y[nmap[k]] * light[1] + normal_z[nmap[k]] * light[2] )
            #printf(" %f ", lum)
            c = chars[<int>(lum*lum / 2)]

            #if fcount-j < 100:
            #    c = b' '

            triangle(
                <int>vbuf_x[faces_a[k]], <int>vbuf_y[faces_a[k]],
                <int>vbuf_x[faces_b[k]], <int>vbuf_y[faces_b[k]],
                <int>vbuf_x[faces_c[k]], <int>vbuf_y[faces_c[k]],
                #b'.'
                c
            )
        #printf("]\n")

        #printf("[")
        #for j from 0 <= j < fcount by 1: # draw faces from depth buf
        #    if dbuf_i[j] in (0,):
        #        printf(" %d(%f)[%f %f - %f %f - %f %f] ",<int>dbuf_i[j], dbuf_d[j],
        #            vbuf_x[faces_a[<int>dbuf_i[j]]], vbuf_y[faces_a[<int>dbuf_i[j]]],
        #            vbuf_x[faces_b[<int>dbuf_i[j]]], vbuf_y[faces_b[<int>dbuf_i[j]]],
        #            vbuf_x[faces_c[<int>dbuf_i[j]]], vbuf_y[faces_c[<int>dbuf_i[j]]])
        #        triangle(
        #            <int>vbuf_x[faces_a[<int>dbuf_i[j]]], <int>vbuf_y[faces_a[<int>dbuf_i[j]]],
        #            <int>vbuf_x[faces_b[<int>dbuf_i[j]]], <int>vbuf_y[faces_b[<int>dbuf_i[j]]],
        #            <int>vbuf_x[faces_c[<int>dbuf_i[j]]], <int>vbuf_y[faces_c[<int>dbuf_i[j]]],
        #            #b'.'
        #            chars[<int>dbuf_i[j]]
        #        )
        #printf("]\n")

        #for j from 0<= j < 36 by 3: 
        #    triangle(
        #        <int>vertex_buf[faces[j]*3],   <int>vertex_buf[faces[j]*3+1],
        #        <int>vertex_buf[faces[j+1]*3], <int>vertex_buf[faces[j+1]*3+1],
        #        <int>vertex_buf[faces[j+2]*3], <int>vertex_buf[faces[j+2]*3+1],
        #        #b'.'
        #        chars[j]
        #    )

        #for j from 0<= j < ecount by 1: #draw edges
        #    line(
        #        <int>vbuf_x[edges_a[j]], <int>vbuf_y[edges_a[j]],
        #        <int>vbuf_x[edges_b[j]], <int>vbuf_y[edges_b[j]],
        #        b'$'
        #    )

        #for j from 0<= j < vcount by 1: # draw points
        #    point(<int>vbuf_x[j], <int>vbuf_y[j], <char>(j+65))

        show()

        t = time()-t
        print(f"{t*10**6:.5f}µs {t*10**3:.5f}ms {1/t:.5f}fps")
        sleep(max(0, 1/60 - t))
        i += 1

except KeyboardInterrupt:
    pass

free(screen)