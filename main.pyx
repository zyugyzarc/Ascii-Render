# distutils: language = c++

import os, sys
from libc.stdlib cimport calloc, free #, system
from libc.stdio cimport printf
from libc.math cimport sin, cos, sqrt
from time import perf_counter as time, sleep
import cython

if "DISPLAY" in os.environ: import pyautogui as pg # to get mouse position

cdef int FPS = 12;
cdef int COLOR = 1; # BOOL : ENUM(0, 1)

if os.environ.get("FPS"):     FPS = <int>int(os.environ["FPS"])
if os.environ.get("NOCOLOR"): COLOR = 0

cdef struct vec:
    float x
    float y
    float z

cdef struct ivec:
    int x
    int y
    int z

cdef int w = <int>(os.get_terminal_size()[0])
cdef int h = <int>(os.get_terminal_size()[1])

printf("\nsize: %d x %d\n", w, h)

cdef char* screen = <char*> calloc( w*h, sizeof(char) );
cdef ivec* color = <ivec*> calloc( w*h, sizeof(ivec) );

cdef void norm( vec& v , float s = 1):
    cdef float m = sqrt( v.x * v.x + v.y * v.y + v.z * v.z )
    v.x = v.x /m *s
    v.y = v.y /m *s
    v.z = v.z /m *s

cdef float mag( vec v ):
    return sqrt( v.x * v.x + v.y * v.y + v.z * v.z )

cdef float dist( vec a, vec b):
    cdef vec v = vec(a.x-b.x, a.y-b.y, a.z-b.z )
    return sqrt( v.x * v.x + v.y * v.y + v.z * v.z )

cdef ivec to_ivec( vec v ):
    return ivec(<int>v.x, <int>v.y, <int>v.z)

cdef vec to_vec( ivec v ):
    return vec(v.x, v.y, v.z)

cdef void clear():

    cdef int i;
    for i from 0 <= i < w*h by 1:
        screen[i] = b" "
        color[i] = ivec(255, 255, 255)
    printf("\x1b[H\x1b[2J")


cdef void show():
    cdef int i;
    for i from 0 <= i < w*h by 1:
        if COLOR==1 and ( color[i].x != 255 or color[i].y != 255 or color[i].z != 255 ):
            printf("\x1b[38;2;%d;%d;%dm%c\x1b[0m",
                color[i].x,
                color[i].y,
                color[i].z,
                screen[i]
            )
        else:
            printf("%c", screen[i])
    printf('\n')

cdef void point(ivec p, char c, ivec col):
    if col.x > 0:
        if 0 < p.y*w + p.x and p.y*w + p.x < h*w:
            screen[p.y*w + p.x] = c
            color[ p.y*w + p.x] = col

#@cython.cdivision(True)
cdef void line(ivec p1, ivec p2, char ch, ivec col):
    
    cdef float m, c
    cdef int i =0

    if p1.x != p2.x:

        m = (p2.y-p1.y)/(p2.x-p1.x)
        c = p1.y - m * p1.x
        
        if p1.x < p2.x:
            for i from p1.x < i < p2.x by 1:
                point( ivec(i, <int>(i*m+c), 0), ch, col )
                

        if p2.x < p1.x:
            for i from p2.x < i < p1.x by 1:
                point( ivec(i, <int>(i*m+c), 0), ch, col )

    if p1.y != p2.y:

        m = (p2.x-p1.x)/(p2.y-p1.y)
        c = p1.x- m*p1.y
        
        if p1.y < p2.y:
            for i from p1.y < i < p2.y by 1:
                point( ivec( <int>(i*m+c), i, 0), ch, col)

        if p2.y < p1.y:
            for i from p2.y < i < p1.y by 1:
                point( ivec( <int>(i*m+c), i, 0), ch, col)

#@cython.cdivision(True)
cdef ivec interpolate_tex(ivec p, ivec p1, ivec p2, ivec p3):

    cdef float m1, c1, m2, c2
    
    #point( p, b"*", ivec(255, 128, 0) )
    #point( p1, b"1", ivec(255, 128, 0) )
    #point( p2, b"2", ivec(255, 128, 0) )
    #point( p3, b"3", ivec(255, 128, 0) )
    
    cdef float s, t

    if p.x == p3.x:
        return ivec(0, 0, 0)

    if p2.x == p1.x:
        return ivec(0, 0, 0)


    m1 = (p2.y-p1.y)/(p2.x-p1.x)
    c1 = p1.y - m1 * p1.x

    m2 = (p3.y - p.y)/(p3.x - p.x)
    c2 = p.y - m2 * p.x
    
    if m1 == m2:
        return ivec(0, 0, 0)

    cdef vec q
    q.x = (c2 - c1)/(m1 - m2)
    q.y = (q.x*m2 + c2) 
    
    if dist(q, to_vec(p3) ) == 0 or dist( to_vec(p1) , to_vec(p2) ) == 0:
        return ivec(0, 0, 0)


    s = dist( to_vec(p1) , q) / dist( to_vec(p1) , to_vec(p2) )
    t = dist(q, to_vec(p) ) / dist(q, to_vec(p3) )
    
    #point( ivec(<int>q.x, <int>q.y, 0), b"%", ivec(255, 255, 0) )
    
    #printf("<%d, %d, 255>\n", s*255, t*255)
    return ivec( <int>(s*255), <int>(t*255), 255 )

    #clear()
    #return ivec(0, 0, 255)

#@cython.cdivision(True)
cdef void triangle(ivec p1, ivec p2, ivec p3, char ch, ivec col):

    cdef int[3] xi = [p1.x, p2.x, p3.x]
    cdef int[3] yi = [p1.y, p2.y, p3.y]
    cdef ivec p4
    cdef float m, c
    cdef int i, j = 0
    cdef int tex = 0

    if col.x < 0:
        tex = 1

    for j from 1<= j < 3 by 1:
        for i from 1<= i < 3 by 1:
            if yi[i-1] > yi[i]:
                yi[i-1], yi[i] = yi[i], yi[i-1]
                xi[i-1], xi[i] = xi[i], xi[i-1]

    p1.x, p2.x, p3.x = xi[0], xi[1], xi[2]
    p1.y, p2.y, p3.y = yi[0], yi[1], yi[2]
    
    cdef float m1, c1, m2, c2
    
    if p1.y == p3.y:
        line(p1, p2, ch, col)
        line(p1, p3, ch, col)
    else:
        m = (p1.x - p3.x)/(p1.y - p3.y)
        c = p3.x - m * p3.y

        p4.x = <int>(m * p2.y + c)
        p4.y = p2.y

        if p1.y == p2.y:
            line(p1, p2, ch, col)
        else:
            # draw flat bottom triangle : p1 | p2 -- p4
            m1 = (p1.x - p2.x)/(p1.y - p2.y)
            c1 = p1.x - m1*p1.y

            m2 = (p1.x - p4.x)/(p1.y - p4.y)
            c2 = p1.x - m2*p1.y

            for i from p1.y <= i <= p2.y by 1:
                for j from <int>(i*m1+c1) <= j <= <int>(i*m2+c2) by 1:
                    if tex:
                        col = interpolate_tex( ivec(j, i, 0), p1, p2, p3)
                        #printf("<%d, %d, %d>  ", col.x, col.y, col.z)
                    point( ivec(j, i, 0), ch, col)

                for j from <int>(i*m2+c2) <= j <= <int>(i*m1+c1) by 1:
                    if tex:
                        col = interpolate_tex( ivec(j, i, 0), p1, p2, p3)
                        #printf("<%d, %d, %d>  ", col.x, col.y, col.z)
                    point( ivec(j, i, 0), ch, col)

        if p2.y == p3.y:
            line(p2, p3, ch, col)
        else:
            # draw flat top tiangle : p2 -- p4 | p3
            m1 = (p3.x - p2.x)/(p3.y - p2.y)
            c1 = p3.x - m1*p3.y

            m2 = (p3.x - p4.x)/(p3.y - p4.y)
            c2 = p3.x - m2*p3.y

            for i from p2.y <= i <= p3.y by 1:
                for j from <int>(i*m1+c1) <= j <= <int>(i*m2+c2) by 1:
                    if tex:
                        col = interpolate_tex( ivec(j, i, 0), p1, p2, p3)
                        #printf("<%d, %d, %d>  ", col.x, col.y, col.z)
                    point( ivec(j, i, 0), ch, col)

                for j from <int>(i*m2+c2) <= j <= <int>(i*m1+c1) by 1:
                    if tex:
                        col = interpolate_tex( ivec(j, i, 0), p1, p2, p3)
                        #printf("<%d, %d, %d>  ", col.x, col.y, col.z)
                    point( ivec(j, i, 0), ch, col)
        printf('\n')

cdef void quad(ivec p1, ivec p2, ivec p3, ivec p4, char c, ivec col):
    triangle(p1, p2, p3, c, col)
    triangle(p2, p3, p4, c, col)


cdef float deg(float x): return 180 * x / 3.14159
cdef float rad(float x): return 3.14159 * x / 180


cdef void mat_mul(float* m1, int a1, int b1, float* m2, int a2, int b2, float* m3):
    
    cdef int i, j, k, l
    cdef float s

    # m1 is a1 x b1 matrix
    # m2 is a2 x b2 matrix

    if b1 != a2: printf("invalid matrix %dx%d x %dx%d\n", a1, b1, a2, b2)

    for i from 0 <= i < a1 by 1:
        for j from 0 <= j < b2 by 1:
            s=0
            for k from 0 <= k < a2 by 1:
                s = s + m1[i*b1+k] * m2[j+k*b2]
            m3[i*b2+j] = s

cdef void rotate_point(vec& p_, vec r):
    
    cdef float p[3]
    p[0], p[1], p[2] = p_.x, p_.y, p_.z

    cdef float Rx[9]
    cdef float Ry[9]
    cdef float Rz[9]
    cdef int i

    Rx[:] = [1,       0,       0, 
             0,  cos(r.x), -sin(r.x),
             0,  sin(r.x),  cos(r.x)]

    Ry[:] = [cos(r.y),  0, -sin(r.y), 
             0     ,  1,       0,
             sin(r.y),  0,  cos(r.y)]

    Rz[:] = [cos(r.z), -sin(r.z),  0, 
             sin(r.z),  cos(r.z),  0,
             0     ,       0,  1]

    cdef float P1[3]
    cdef float P2[3]

    mat_mul(Rx, 3, 3, p , 3, 1, P1)
    mat_mul(Ry, 3, 3, P1, 3, 1, P2)
    mat_mul(Rz, 3, 3, P2, 3, 1, p )

    p_.x, p_.y, p_.z = p[0], p[1], p[2]

cdef void move_point(vec& p, float x, float y, float z):
    p.x += x
    p.y += y
    p.z += z

cdef vec project_point(vec& p, float sx, float sy):

    cdef float distance = 5.0
    cdef float z = 1/(distance - p.z)
    p.x = p.x * z * sx + w/2
    p.y = p.y * z * sy + h/2
    p.z = z



cdef int color_char(ivec& c):
    
    if c.x < 0:
        return 10

    cdef int m
    m = ( c.x if c.x > c.y else c.y )
    m = ( m   if m   > c.z else c.z )

    c.x = <int> ( 200/m * c.x ) + <int>(m * 50/255 * c.x/255)
    c.y = <int> ( 200/m * c.y ) + <int>(m * 50/255 * c.y/255)
    c.z = <int> ( 200/m * c.z ) + <int>(m * 50/255 * c.z/255)

    return m


#clear()


#cpdef class Mesh:
cdef class Mesh:

    #cdef vec points[512];
    #cdef vec normals[1024];
    #cdef ivec faces[1024];
    #cdef int fmat[1024];
    #cdef int nmap[1024];

    #cdef vec vbuf[512];
    #cdef float dbuf[1024];
    #cdef int dbuf_idx[1024];

    #for large models
    cdef vec points[2048];
    cdef vec normals[4096];
    cdef ivec faces[4096];
    cdef int fmat[4096];
    cdef int nmap[4096];

    cdef vec vbuf[2048];
    cdef float dbuf[4096];
    cdef int dbuf_idx[4096];
    

    cdef ivec pallete[16];
    cdef int vcount
    cdef int fcount

    cdef int i
    
    cdef public vec pos;
    cdef public vec rot;
    

    cdef vec apply_transform(self, vec& p):
        
        rotate_point(p, self.rot)
        return p

    cdef char chars[37]

    cdef void project_points(self):
        
        cdef int j
        cdef vec p
        cdef int l = ( w if w < h else h ) * 4
        for j from 0<= j < self.vcount by 1:

            p = self.points[j]
            self.apply_transform(p)
            project_point(p, l, l/2)
            self.vbuf[j] = p

    cdef void order_depth(self):

        cdef ivec face
        cdef int j, k
        for j from 0<= j < self.fcount by 1: # fill depth buf
                
                face = self.faces[j]
                self.dbuf[j] = (self.vbuf[ face.x ].z + 
                                self.vbuf[ face.y ].z +
                                self.vbuf[ face.z ].z )/3
                self.dbuf_idx[j] = j

        
        for j from 0<= j < self.fcount by 1: # sort depth buf
            for k from 1<= k < self.fcount by 1:
                if self.dbuf[k-1] > self.dbuf[k]:
                    self.dbuf[k-1], self.dbuf[k] = self.dbuf[k], self.dbuf[k-1]
                    self.dbuf_idx[k-1], self.dbuf_idx[k] = self.dbuf_idx[k], self.dbuf_idx[k-1]

    cpdef void render(self):
        
        self.project_points()
        self.order_depth()

        cdef int j, k
        cdef vec n, l
        cdef float lum
        cdef ivec col
        cdef char c
        
        for j from 0 <= j < self.fcount by 1:
            
            k = self.dbuf_idx[j]
            n = self.normals[ self.nmap[k] ]
            l = vec(7, -7, 5)
            norm(l, 5)
            
            self.apply_transform(n)
            self.apply_transform(l)

            # get color from materials (mtl file) 
            
            # color transform:
            # color in (RGB) -> (HSV)
            # V channel => lum, color (HS V=100%) -> (RGB)
            col = self.pallete[ self.fmat[k] ]
            if self.fmat[k] == 2:
                #col = ivec(32, 64, 128)
                col = ivec( -1, 0, 0)

            lum = ( n.x * l.x + n.y * l.y + n.z * l.z )
            lum = lum * lum * (255 - color_char(col) )/255 + 1
            lum = lum if lum > 0 else 0
            lum = lum if lum < 36 else 36
            c = self.chars[<int>lum]

            # Draw face

            triangle(
                ivec( <int>self.vbuf[ self.faces[k].x ].x, <int>self.vbuf[ self.faces[k].x ].y, 0),
                ivec( <int>self.vbuf[ self.faces[k].y ].x, <int>self.vbuf[ self.faces[k].y ].y, 0),
                ivec( <int>self.vbuf[ self.faces[k].z ].x, <int>self.vbuf[ self.faces[k].z ].y, 0),
                #b'.'
                c, col
            )

            # Draw Edge

            #line(
            #    ivec( <int>self.vbuf[ self.faces[k].x ].x, <int>self.vbuf[ self.faces[k].x ].y, 0),
            #    ivec( <int>self.vbuf[ self.faces[k].y ].x, <int>self.vbuf[ self.faces[k].y ].y, 0),
            #    b'#', ivec( 0, 255, 255 )
            #)
            #line(
            #    ivec( <int>self.vbuf[ self.faces[k].y ].x, <int>self.vbuf[ self.faces[k].y ].y, 0),
            #    ivec( <int>self.vbuf[ self.faces[k].z ].x, <int>self.vbuf[ self.faces[k].z ].y, 0),
            #    b'#', ivec( 0, 255, 255 )
            #)
            #line(
            #    ivec( <int>self.vbuf[ self.faces[k].x ].x, <int>self.vbuf[ self.faces[k].x ].y, 0),
            #    ivec( <int>self.vbuf[ self.faces[k].z ].x, <int>self.vbuf[ self.faces[k].z ].y, 0),
            #    b'#', ivec( 0, 255, 255 )
            #)

            # Draw Vertex
            
            #col = ivec(0, 255, 255)
            #point( ivec( <int>self.vbuf[self.faces[k].x].x, <int>self.vbuf[self.faces[k].x].y, 0 ), b'@', col )
            #point( ivec( <int>self.vbuf[self.faces[k].y].x, <int>self.vbuf[self.faces[k].y].y, 0 ), b'@', col ); col = ivec(0, 192, 255)
            #point( ivec( <int>self.vbuf[self.faces[k].z].x, <int>self.vbuf[self.faces[k].z].y, 0 ), b'@', col )


        show()

        self.i += 1
        

    #Load OBJ File (python)
    def __init__(self, FILENAME):
        
        cdef int i
        for i from 0<=i<37 by 1: self.chars[i] = ".:`\'-,;~_!\"?c\\^<>|=sr1Jo*(C)utia3zLvey75jST{lx}IfY]qp9n0G62Vk8UXhZ4bgdPEKA$wQm&#HDR@WNBM"[i]

        cdef int _pi = 0, _fi = 0, _ni = 0, _mi = 0
        mtlfile = ""
        with open(FILENAME) as f:
            for i_ in f.readlines():
                try:
                    if i_.startswith('vn'):
                        data = i_[3:].strip().replace(' ',',')
                        #_normals.append( eval(data) )
                        x, y, z = eval(data)
                        self.normals[_ni] = vec(<float>x, <float>y, <float>z)
                        _ni += 1

                    if i_.startswith("vt"):
                        pass
    
                    elif i_.startswith('v'):
                        data = i_[2:].strip().replace(' ',',')
                        #_points.append( eval(data) )
                        x, y, z = eval(data)
                        self.points[_pi] = vec(<float>x, <float>y, <float>z)
                        _pi += 1
    

                    elif i_.startswith('f'): 
                        data = i_[2:].strip().replace(' ',',').replace('/',',').replace(',,',',')
                        data = [j_-1 for j_ in eval(data)]
                        #_faces.append( data[::2] )
                        #_nmap.append(data[1])

                        assert len(data) > 3, RuntimeError("No Normals in OBJ File")
                        
                        if len(data) == 6: # [ v//n ] * 3
                            self.nmap[_fi] = data[1]
                            self.faces[_fi] = ivec( data[0], data[2], data[4] )
                        if len(data) == 9: # [ v/t/n ] * 3
                            self.nmap[_fi] = data[2]
                            self.faces[_fi] = ivec( data[0], data[3], data[6] )

                        self.fmat[_fi] = _mi
                        _fi += 1

                    elif i_.startswith("mtllib"):
                        mtlfile = i_[7:]

                    elif i_.startswith("usemtl"):
                        data = i_[6:]
                        _mi += 1
                        
                        x, y, z = self.load_mtl( FILENAME, mtlfile, data )
                        self.pallete[_mi] = ivec(x, y, z)

                except Exception as e:
                    print( "in ", i_, ",", "(",FILENAME, mtlfile,")")
                    raise e
        

        self.vcount = <int>_pi
        self.fcount = <int>_fi

        printf("loaded model: %d verts, %d tris\n", self.vcount, self.fcount)
    
    def load_mtl(self, ofile, file, mat):
        with open( os.path.join(os.path.dirname(ofile) ,file.strip() ) ) as f:
            f = f.read()
        f = f[f.find(mat):]
        f = f[f.find("Kd"):]
        f = f[3:f.find("\n")]
        f = eval(f.replace(' ', ','))
        r, g, b = f
        r *= 255; g*= 255; b*= 255
        return int(r), int(g), int(b)



cpdef main():
    m = Mesh( sys.argv[1] )
    input('[press enter to continue]')
    m.rot.x = rad(100)
    while True:
        t = time()
        
        m.rot.y += 1/FPS * 0.5

        # for mouse controls
        #p = pg.position()
        #m.rot.x = -p[1]/100
        #m.rot.y = -p[0]/100
        
        clear()
        m.render()
    
        t = time()-t
        print(f"{t*1_000_000} µs {t*1000:.4f} ms {1/t:.3f} fps")
        sleep( max( 0, 1/FPS-t ) )

main()


free(screen)
free(color)
#"""
