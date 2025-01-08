import bpy
import bpy_extras
from mathutils import Vector
import inspect

from bpy_extras.object_utils import world_to_camera_view

## Deselect mesh polygons and vertices
#def DeselectEdgesAndPolygons( obj ):
#    for p in obj.data.polygons:
#        p.select = False
#    for e in obj.data.edges:
#        e.select = False

## Get context elements: scene, camera and mesh
#scene = bpy.context.scene
#cam = bpy.data.objects['Camera']
#obj = bpy.data.objects['Plane']

##renderer an image and save to F drive
#scene.render.image_settings.file_format = 'PNG'
#scene.render.filepath = "/home/habeeb/image100.png"
#bpy.ops.render.render(write_still = 1)

#width =  scene.render.resolution_x 
#height = scene.render.resolution_y

## Threshold to test if ray cast corresponds to the original vertex
#limit = 0.1

## Deselect mesh elements
#DeselectEdgesAndPolygons( obj )

## In world coordinates, get a bvh tree and vertices
#mWorld = obj.matrix_world
#vertices = [mWorld @ v.co for v in obj.data.vertices]

#print( '-------------------' )
#print("vertices ",vertices)

#for i, v in enumerate( vertices ):
#    print(v)
#    # Get the 2D projection of the vertex
#    co2D = world_to_camera_view( scene, cam, v )
#    print("co2d",co2D)
#    cubeMesh=bpy.ops.mesh.primitive_cube_add(location=(v))
#    bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))
#    cubeObject=bpy.context.object

#    # If inside the camera view
#    if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0 and co2D.z >0: 
#        # Try a ray cast, in order to test the vertex visibility from the camera
#        location= scene.ray_cast(bpy.context.view_layer, cam.location, (v - cam.location).normalized() )
#        # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
#        if location[0] and (v - location[1]).length < limit:
#            print("%.2f %.2f" %(co2D.x*width,co2D.y*height))
#    bpy.data.objects.remove(cubeObject,do_unlink= True)

##############################################

for obj in bpy.context.selected_objects:
    #current_obj = bpy.context.active_object 
    current_obj =obj
    

    verts_local = [v.co for v in current_obj.data.vertices.values()]
    verts_world = [current_obj.matrix_world @ v_local for v_local in verts_local]

#    print("=====================")
    print("//",obj.name)
    #print("World coordinates")
    #print(verts_world) 
#    print("vertex coordinates")    
    for i, vert in enumerate(verts_world):
        print(" {{ {v[1]}, {v[2]}, {v[0]} }}, ".format(i=i, v=vert))
        #print(" {v[1]}, {v[2]}, {v[0]} , ".format(i=i, v=vert))
#    print("face vertex indices")
    for i, face in enumerate(current_obj.data.polygons):
        verts_indices = face.vertices[:]
        print(" //{{ {v_i[0]}, {v_i[1]}, {v_i[2]} }},".format(i=i, v_i=verts_indices))
        #print(" ")

print("..........///////............................")

#if bpy.context.selected_objects != []:
#    for obj in bpy.context.selected_objects:
#        print(obj.name, obj, obj.type)
##        if obj.type == 'MESH': 
#            print(obj.name)




#scene = bpy.context.scene
#obj = bpy.context.active_object    # object you want the coordinates of
#vertices = obj.data.vertices   # you will get the coordinates of its vertices
#camera = bpy.data.objects["Camera"]

#for v in vertices:
#    # local to global coordinates
#    co = v.co @ obj.matrix_world
#    # calculate 2d image coordinates
#    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, co)
#    render_scale = scene.render.resolution_percentage / 100
#    render_size = (
#        int(scene.render.resolution_x * render_scale),
#        int(scene.render.resolution_y * render_scale),
#    )

#    # this is the result
#    pixel_coords = (co_2d.x * render_size[0],
#                    co_2d.y * render_size[1])

#    print("Pixel Coords:", (
#          round(pixel_coords[0]),
#          round(pixel_coords[1]),
#    ))

#print("Render scale : ",scene.render.resolution_percentage)
#print("Screen resolution x ", scene.render.resolution_x)
#print("===============")

##############################################################################


#def project_3d_point(camera: bpy.types.Object,
#                     p: Vector,
#                     render: bpy.types.RenderSettings = bpy.context.scene.render) -> Vector:
#    """
#    Given a camera and its projection matrix M;
#    given p, a 3d point to project:

#    Compute P’ = M * P
#    P’= (x’, y’, z’, w')

#    Ignore z'
#    Normalize in:
#    x’’ = x’ / w’
#    y’’ = y’ / w’

#    x’’ is the screen coordinate in normalised range -1 (left) +1 (right)
#    y’’ is the screen coordinate in  normalised range -1 (bottom) +1 (top)

#    :param camera: The camera for which we want the projection
#    :param p: The 3D point to project
#    :param render: The render settings associated to the scene.
#    :return: The 2D projected point in normalized range [-1, 1] (left to right, bottom to top)
#    """

#    if camera.type != 'CAMERA':
#        raise Exception("Object {} is not a camera.".format(camera.name))

#    if len(p) != 3:
#        raise Exception("Vector {} is not three-dimensional".format(p))

#    # Get the two components to calculate M
#    modelview_matrix = camera.matrix_world.inverted()
#    print("Model view matrix ",modelview_matrix)
#    projection_matrix = camera.calc_matrix_camera(
#        bpy.data.scenes["Scene"].view_layers["View Layer"].depsgraph,
#        x = render.resolution_x,
#        y = render.resolution_y,
#        scale_x = render.pixel_aspect_x,
#        scale_y = render.pixel_aspect_y,
#    )
#    
#    print("Projection Matrix ",projection_matrix)

#    # print(projection_matrix * modelview_matrix)

#    # Compute P’ = M * P
#    p1 = projection_matrix @ modelview_matrix @ Vector((p.x, p.y, p.z, 1))

#    # Normalize in: x’’ = x’ / w’, y’’ = y’ / w’
#    p2 = Vector(((p1.x/p1.w, p1.y/p1.w)))

#    return p2

#camera = bpy.data.objects['Camera']  # or bpy.context.active_object
#render = bpy.context.scene.render

##P = Vector((-0.002170146, 0.409979939, 0.162410125))

#P = Vector((-4.0, 0.0, -200.0))

#print("Projecting point {} for camera '{:s}' into resolution {:d}x{:d}..."
#      .format(P, camera.name, render.resolution_x, render.resolution_y))

#proj_p = project_3d_point(camera=camera, p=P, render=render)
#print("Projected point (homogeneous coords): {}.".format(proj_p))

#proj_p_pixels = Vector(((render.resolution_x-1) * (proj_p.x + 1) / 2, (render.resolution_y - 1) * (proj_p.y - 1) / (-2)))
#print("Projected point (pixel coords): {}.".format(proj_p_pixels))

#print("Done.")