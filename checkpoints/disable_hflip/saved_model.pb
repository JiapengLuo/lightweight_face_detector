ш4
Ё-і,
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
+
Ceil
x"T
y"T"
Ttype:
2
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

н
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	

FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%Зб8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
!
LoopCond	
input


output


Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp

NonMaxSuppressionV2

boxes"T
scores"T
max_output_size
iou_threshold
selected_indices"
Ttype0:
2
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
q
ResizeBilinear
images"T
size
resized_images"
Ttype:
2
	"
align_cornersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
9
TensorArraySizeV3

handle
flow_in
size
о
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*1.12.02v1.12.0-0-ga6d8ffae098ю!

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
shape: *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step

input_imagesPlaceholder*-
shape$:"џџџџџџџџџџџџџџџџџџ*
dtype0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ
`
image_pyramid/assert_rank/rankConst*
value	B :*
dtype0*
_output_shapes
: 
[
image_pyramid/assert_rank/ShapeShapeinput_images*
T0*
_output_shapes
:
P
Himage_pyramid/assert_rank/assert_type/statically_determined_correct_typeNoOp
A
9image_pyramid/assert_rank/static_checks_determined_all_okNoOp
Џ
image_pyramid/IdentityIdentityinput_images:^image_pyramid/assert_rank/static_checks_determined_all_ok*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
T0
`
image_pyramid/TensorArray/sizeConst*
value	B : *
dtype0*
_output_shapes
: 
и
image_pyramid/TensorArrayTensorArrayV3image_pyramid/TensorArray/size*
_output_shapes

:: *5
element_shape$:"џџџџџџџџџџџџџџџџџџ*
clear_after_read( *
dynamic_size(*
dtype0
Y
image_pyramid/ToFloat/xConst*
value	B :*
dtype0*
_output_shapes
: 
f
image_pyramid/ToFloatCastimage_pyramid/ToFloat/x*
_output_shapes
: *

DstT0*

SrcT0
m
+image_pyramid/img_shape/assert_rank_in/rankConst*
value	B :*
dtype0*
_output_shapes
: 
o
-image_pyramid/img_shape/assert_rank_in/rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
r
,image_pyramid/img_shape/assert_rank_in/ShapeShapeimage_pyramid/Identity*
T0*
_output_shapes
:
]
Uimage_pyramid/img_shape/assert_rank_in/assert_type/statically_determined_correct_typeNoOp
_
Wimage_pyramid/img_shape/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp
N
Fimage_pyramid/img_shape/assert_rank_in/static_checks_determined_all_okNoOp
Ї
image_pyramid/img_shape/RankConstG^image_pyramid/img_shape/assert_rank_in/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 
Њ
image_pyramid/img_shape/Equal/yConstG^image_pyramid/img_shape/assert_rank_in/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 

image_pyramid/img_shape/EqualEqualimage_pyramid/img_shape/Rankimage_pyramid/img_shape/Equal/y*
T0*
_output_shapes
: 

#image_pyramid/img_shape/cond/SwitchSwitchimage_pyramid/img_shape/Equalimage_pyramid/img_shape/Equal*
T0
*
_output_shapes
: : 
y
%image_pyramid/img_shape/cond/switch_tIdentity%image_pyramid/img_shape/cond/Switch:1*
T0
*
_output_shapes
: 
w
%image_pyramid/img_shape/cond/switch_fIdentity#image_pyramid/img_shape/cond/Switch*
T0
*
_output_shapes
: 
p
$image_pyramid/img_shape/cond/pred_idIdentityimage_pyramid/img_shape/Equal*
T0
*
_output_shapes
: 
Ц
"image_pyramid/img_shape/cond/ShapeShape+image_pyramid/img_shape/cond/Shape/Switch:1G^image_pyramid/img_shape/assert_rank_in/static_checks_determined_all_ok*
T0*
_output_shapes
:

)image_pyramid/img_shape/cond/Shape/SwitchSwitchimage_pyramid/Identity$image_pyramid/img_shape/cond/pred_id*
T0*)
_class
loc:@image_pyramid/Identity*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ
ы
0image_pyramid/img_shape/cond/strided_slice/stackConstG^image_pyramid/img_shape/assert_rank_in/static_checks_determined_all_ok&^image_pyramid/img_shape/cond/switch_t*
_output_shapes
:*
valueB: *
dtype0
э
2image_pyramid/img_shape/cond/strided_slice/stack_1ConstG^image_pyramid/img_shape/assert_rank_in/static_checks_determined_all_ok&^image_pyramid/img_shape/cond/switch_t*
dtype0*
_output_shapes
:*
valueB:
э
2image_pyramid/img_shape/cond/strided_slice/stack_2ConstG^image_pyramid/img_shape/assert_rank_in/static_checks_determined_all_ok&^image_pyramid/img_shape/cond/switch_t*
valueB:*
dtype0*
_output_shapes
:
М
*image_pyramid/img_shape/cond/strided_sliceStridedSlice"image_pyramid/img_shape/cond/Shape0image_pyramid/img_shape/cond/strided_slice/stack2image_pyramid/img_shape/cond/strided_slice/stack_12image_pyramid/img_shape/cond/strided_slice/stack_2*
_output_shapes
:*
Index0*
T0*

begin_mask
Ш
$image_pyramid/img_shape/cond/Shape_1Shape+image_pyramid/img_shape/cond/Shape_1/SwitchG^image_pyramid/img_shape/assert_rank_in/static_checks_determined_all_ok*
_output_shapes
:*
T0

+image_pyramid/img_shape/cond/Shape_1/SwitchSwitchimage_pyramid/Identity$image_pyramid/img_shape/cond/pred_id*
T0*)
_class
loc:@image_pyramid/Identity*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ
э
2image_pyramid/img_shape/cond/strided_slice_1/stackConstG^image_pyramid/img_shape/assert_rank_in/static_checks_determined_all_ok&^image_pyramid/img_shape/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
я
4image_pyramid/img_shape/cond/strided_slice_1/stack_1ConstG^image_pyramid/img_shape/assert_rank_in/static_checks_determined_all_ok&^image_pyramid/img_shape/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
я
4image_pyramid/img_shape/cond/strided_slice_1/stack_2ConstG^image_pyramid/img_shape/assert_rank_in/static_checks_determined_all_ok&^image_pyramid/img_shape/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
Д
,image_pyramid/img_shape/cond/strided_slice_1StridedSlice$image_pyramid/img_shape/cond/Shape_12image_pyramid/img_shape/cond/strided_slice_1/stack4image_pyramid/img_shape/cond/strided_slice_1/stack_14image_pyramid/img_shape/cond/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
:
Е
"image_pyramid/img_shape/cond/MergeMerge,image_pyramid/img_shape/cond/strided_slice_1*image_pyramid/img_shape/cond/strided_slice*
T0*
N*
_output_shapes

:: 
|
image_pyramid/img_shape/CastCast"image_pyramid/img_shape/cond/Merge*

SrcT0*
_output_shapes
:*

DstT0
X
image_pyramid/mul/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
p
image_pyramid/mulMulimage_pyramid/mul/ximage_pyramid/img_shape/Cast*
T0*
_output_shapes
:
o
image_pyramid/truedivRealDivimage_pyramid/mulimage_pyramid/ToFloat*
T0*
_output_shapes
:
V
image_pyramid/CeilCeilimage_pyramid/truediv*
T0*
_output_shapes
:
j
image_pyramid/mul_1Mulimage_pyramid/Ceilimage_pyramid/ToFloat*
T0*
_output_shapes
:
[
image_pyramid/while/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
Ќ
image_pyramid/while/EnterEnterimage_pyramid/while/Const*
T0*
parallel_iterations*
_output_shapes
: *1

frame_name#!image_pyramid/while/while_context
А
image_pyramid/while/Enter_1Enterimage_pyramid/TensorArray:1*
_output_shapes
: *1

frame_name#!image_pyramid/while/while_context*
T0*
parallel_iterations
Ќ
image_pyramid/while/Enter_2Enterimage_pyramid/mul_1*
parallel_iterations*
_output_shapes
:*1

frame_name#!image_pyramid/while/while_context*
T0

image_pyramid/while/MergeMergeimage_pyramid/while/Enter!image_pyramid/while/NextIteration*
N*
_output_shapes
: : *
T0

image_pyramid/while/Merge_1Mergeimage_pyramid/while/Enter_1#image_pyramid/while/NextIteration_1*
T0*
N*
_output_shapes
: : 

image_pyramid/while/Merge_2Mergeimage_pyramid/while/Enter_2#image_pyramid/while/NextIteration_2*
T0*
N*
_output_shapes

:: 

image_pyramid/while/Const_1Const^image_pyramid/while/Merge*
valueB: *
dtype0*
_output_shapes
:
y
image_pyramid/while/MinMinimage_pyramid/while/Merge_2image_pyramid/while/Const_1*
T0*
_output_shapes
: 
~
image_pyramid/while/Greater/yConst^image_pyramid/while/Merge*
valueB
 *  A*
dtype0*
_output_shapes
: 

image_pyramid/while/GreaterGreaterimage_pyramid/while/Minimage_pyramid/while/Greater/y*
_output_shapes
: *
T0
]
image_pyramid/while/LoopCondLoopCondimage_pyramid/while/Greater*
_output_shapes
: 
Ў
image_pyramid/while/SwitchSwitchimage_pyramid/while/Mergeimage_pyramid/while/LoopCond*
_output_shapes
: : *
T0*,
_class"
 loc:@image_pyramid/while/Merge
Д
image_pyramid/while/Switch_1Switchimage_pyramid/while/Merge_1image_pyramid/while/LoopCond*
T0*.
_class$
" loc:@image_pyramid/while/Merge_1*
_output_shapes
: : 
М
image_pyramid/while/Switch_2Switchimage_pyramid/while/Merge_2image_pyramid/while/LoopCond* 
_output_shapes
::*
T0*.
_class$
" loc:@image_pyramid/while/Merge_2
g
image_pyramid/while/IdentityIdentityimage_pyramid/while/Switch:1*
T0*
_output_shapes
: 
k
image_pyramid/while/Identity_1Identityimage_pyramid/while/Switch_1:1*
T0*
_output_shapes
: 
o
image_pyramid/while/Identity_2Identityimage_pyramid/while/Switch_2:1*
T0*
_output_shapes
:

image_pyramid/while/truedivRealDivimage_pyramid/while/Identity_2!image_pyramid/while/truediv/Enter*
T0*
_output_shapes
:
У
!image_pyramid/while/truediv/EnterEnterimage_pyramid/ToFloat*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *1

frame_name#!image_pyramid/while/while_context
b
image_pyramid/while/CeilCeilimage_pyramid/while/truediv*
T0*
_output_shapes
:

image_pyramid/while/mulMulimage_pyramid/while/Ceil!image_pyramid/while/truediv/Enter*
T0*
_output_shapes
:
p
image_pyramid/while/ToInt32Castimage_pyramid/while/mul*
_output_shapes
:*

DstT0*

SrcT0
г
"image_pyramid/while/ResizeBilinearResizeBilinear(image_pyramid/while/ResizeBilinear/Enterimage_pyramid/while/ToInt32*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
align_corners(*
T0
э
(image_pyramid/while/ResizeBilinear/EnterEnterimage_pyramid/Identity*
T0*
is_constant(*
parallel_iterations*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*1

frame_name#!image_pyramid/while/while_context
Ц
7image_pyramid/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3=image_pyramid/while/TensorArrayWrite/TensorArrayWriteV3/Enterimage_pyramid/while/Identity"image_pyramid/while/ResizeBilinearimage_pyramid/while/Identity_1*
T0*5
_class+
)'loc:@image_pyramid/while/ResizeBilinear*
_output_shapes
: 

=image_pyramid/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterimage_pyramid/TensorArray*
parallel_iterations*
is_constant(*
_output_shapes
:*1

frame_name#!image_pyramid/while/while_context*
T0*5
_class+
)'loc:@image_pyramid/while/ResizeBilinear
z
image_pyramid/while/add/yConst^image_pyramid/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
x
image_pyramid/while/addAddimage_pyramid/while/Identityimage_pyramid/while/add/y*
_output_shapes
: *
T0

image_pyramid/while/mul_1/yConst^image_pyramid/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *   ?
{
image_pyramid/while/mul_1Mulimage_pyramid/while/mulimage_pyramid/while/mul_1/y*
T0*
_output_shapes
:
l
!image_pyramid/while/NextIterationNextIterationimage_pyramid/while/add*
T0*
_output_shapes
: 

#image_pyramid/while/NextIteration_1NextIteration7image_pyramid/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
t
#image_pyramid/while/NextIteration_2NextIterationimage_pyramid/while/mul_1*
T0*
_output_shapes
:
]
image_pyramid/while/ExitExitimage_pyramid/while/Switch*
T0*
_output_shapes
: 
a
image_pyramid/while/Exit_1Exitimage_pyramid/while/Switch_1*
T0*
_output_shapes
: 
e
image_pyramid/while/Exit_2Exitimage_pyramid/while/Switch_2*
_output_shapes
:*
T0
X
zerosConst*
valueB
 *
dtype0* 
_output_shapes
:
 
R
zeros_1Const*
valueB
 *
dtype0*
_output_shapes

: 
M
while/ConstConst*
value	B : *
dtype0*
_output_shapes
: 

while/EnterEnterwhile/Const*
_output_shapes
: *#

frame_namewhile/while_context*
T0*
parallel_iterations

while/Enter_1Enterzeros*+
_output_shapes
:џџџџџџџџџ*#

frame_namewhile/while_context*
T0*
parallel_iterations

while/Enter_2Enterzeros_1*
T0*
parallel_iterations*'
_output_shapes
:џџџџџџџџџ*#

frame_namewhile/while_context
b
while/MergeMergewhile/Enterwhile/NextIteration*
T0*
N*
_output_shapes
: : 
}
while/Merge_1Mergewhile/Enter_1while/NextIteration_1*-
_output_shapes
:џџџџџџџџџ: *
T0*
N
y
while/Merge_2Mergewhile/Enter_2while/NextIteration_2*
T0*
N*)
_output_shapes
:џџџџџџџџџ: 
[
while/add/yConst^while/Merge*
_output_shapes
: *
value	B :*
dtype0
K
	while/addAddwhile/Mergewhile/add/y*
T0*
_output_shapes
: 

while/TensorArraySizeV3TensorArraySizeV3while/TensorArraySizeV3/Enterwhile/TensorArraySizeV3/Enter_1^while/Merge*
_output_shapes
: 
Й
while/TensorArraySizeV3/EnterEnterimage_pyramid/TensorArray*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context*
T0*
is_constant(
И
while/TensorArraySizeV3/Enter_1Enterimage_pyramid/while/Exit_1*
_output_shapes
: *#

frame_namewhile/while_context*
T0*
is_constant(*
parallel_iterations
W

while/LessLess	while/addwhile/TensorArraySizeV3*
T0*
_output_shapes
: 
>
while/LoopCondLoopCond
while/Less*
_output_shapes
: 
v
while/SwitchSwitchwhile/Mergewhile/LoopCond*
T0*
_class
loc:@while/Merge*
_output_shapes
: : 
І
while/Switch_1Switchwhile/Merge_1while/LoopCond*
T0* 
_class
loc:@while/Merge_1*B
_output_shapes0
.:џџџџџџџџџ:џџџџџџџџџ

while/Switch_2Switchwhile/Merge_2while/LoopCond*
T0* 
_class
loc:@while/Merge_2*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
K
while/IdentityIdentitywhile/Switch:1*
_output_shapes
: *
T0
d
while/Identity_1Identitywhile/Switch_1:1*
T0*+
_output_shapes
:џџџџџџџџџ
`
while/Identity_2Identitywhile/Switch_2:1*
T0*'
_output_shapes
:џџџџџџџџџ
У
while/TensorArrayReadV3TensorArrayReadV3while/TensorArraySizeV3/Enterwhile/Identitywhile/TensorArraySizeV3/Enter_1*
dtype0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ
v
#while/img_shape/assert_rank_in/rankConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
x
%while/img_shape/assert_rank_in/rank_1Const^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
k
$while/img_shape/assert_rank_in/ShapeShapewhile/TensorArrayReadV3*
T0*
_output_shapes
:
f
Mwhile/img_shape/assert_rank_in/assert_type/statically_determined_correct_typeNoOp^while/Identity
h
Owhile/img_shape/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp^while/Identity
W
>while/img_shape/assert_rank_in/static_checks_determined_all_okNoOp^while/Identity

while/img_shape/RankConst?^while/img_shape/assert_rank_in/static_checks_determined_all_ok*
dtype0*
_output_shapes
: *
value	B :

while/img_shape/Equal/yConst?^while/img_shape/assert_rank_in/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 
n
while/img_shape/EqualEqualwhile/img_shape/Rankwhile/img_shape/Equal/y*
T0*
_output_shapes
: 
v
while/img_shape/cond/SwitchSwitchwhile/img_shape/Equalwhile/img_shape/Equal*
T0
*
_output_shapes
: : 
i
while/img_shape/cond/switch_tIdentitywhile/img_shape/cond/Switch:1*
T0
*
_output_shapes
: 
g
while/img_shape/cond/switch_fIdentitywhile/img_shape/cond/Switch*
T0
*
_output_shapes
: 
`
while/img_shape/cond/pred_idIdentitywhile/img_shape/Equal*
T0
*
_output_shapes
: 
Ў
while/img_shape/cond/ShapeShape#while/img_shape/cond/Shape/Switch:1?^while/img_shape/assert_rank_in/static_checks_determined_all_ok*
T0*
_output_shapes
:
ѕ
!while/img_shape/cond/Shape/SwitchSwitchwhile/TensorArrayReadV3while/img_shape/cond/pred_id*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ*
T0**
_class 
loc:@while/TensorArrayReadV3
г
(while/img_shape/cond/strided_slice/stackConst?^while/img_shape/assert_rank_in/static_checks_determined_all_ok^while/img_shape/cond/switch_t*
_output_shapes
:*
valueB: *
dtype0
е
*while/img_shape/cond/strided_slice/stack_1Const?^while/img_shape/assert_rank_in/static_checks_determined_all_ok^while/img_shape/cond/switch_t*
valueB:*
dtype0*
_output_shapes
:
е
*while/img_shape/cond/strided_slice/stack_2Const?^while/img_shape/assert_rank_in/static_checks_determined_all_ok^while/img_shape/cond/switch_t*
valueB:*
dtype0*
_output_shapes
:

"while/img_shape/cond/strided_sliceStridedSlicewhile/img_shape/cond/Shape(while/img_shape/cond/strided_slice/stack*while/img_shape/cond/strided_slice/stack_1*while/img_shape/cond/strided_slice/stack_2*

begin_mask*
T0*
Index0*
_output_shapes
:
А
while/img_shape/cond/Shape_1Shape#while/img_shape/cond/Shape_1/Switch?^while/img_shape/assert_rank_in/static_checks_determined_all_ok*
T0*
_output_shapes
:
ї
#while/img_shape/cond/Shape_1/SwitchSwitchwhile/TensorArrayReadV3while/img_shape/cond/pred_id*
T0**
_class 
loc:@while/TensorArrayReadV3*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ
е
*while/img_shape/cond/strided_slice_1/stackConst?^while/img_shape/assert_rank_in/static_checks_determined_all_ok^while/img_shape/cond/switch_f*
dtype0*
_output_shapes
:*
valueB:
з
,while/img_shape/cond/strided_slice_1/stack_1Const?^while/img_shape/assert_rank_in/static_checks_determined_all_ok^while/img_shape/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
з
,while/img_shape/cond/strided_slice_1/stack_2Const?^while/img_shape/assert_rank_in/static_checks_determined_all_ok^while/img_shape/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:

$while/img_shape/cond/strided_slice_1StridedSlicewhile/img_shape/cond/Shape_1*while/img_shape/cond/strided_slice_1/stack,while/img_shape/cond/strided_slice_1/stack_1,while/img_shape/cond/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
:

while/img_shape/cond/MergeMerge$while/img_shape/cond/strided_slice_1"while/img_shape/cond/strided_slice*
T0*
N*
_output_shapes

:: 
b
while/truediv/yConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
j
while/truediv/CastCastwhile/img_shape/cond/Merge*
_output_shapes
:*

DstT0*

SrcT0
]
while/truediv/Cast_1Castwhile/truediv/y*

SrcT0*
_output_shapes
: *

DstT0
g
while/truedivRealDivwhile/truediv/Castwhile/truediv/Cast_1*
_output_shapes
:*
T0
X
while/ToInt32Castwhile/truediv*

SrcT0*
_output_shapes
:*

DstT0
І
while/ResizeBilinearResizeBilinearwhile/TensorArrayReadV3while/ToInt32*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
align_corners(*
T0
x
%while/img_shape_1/assert_rank_in/rankConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
z
'while/img_shape_1/assert_rank_in/rank_1Const^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
m
&while/img_shape_1/assert_rank_in/ShapeShapewhile/TensorArrayReadV3*
T0*
_output_shapes
:
h
Owhile/img_shape_1/assert_rank_in/assert_type/statically_determined_correct_typeNoOp^while/Identity
j
Qwhile/img_shape_1/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp^while/Identity
Y
@while/img_shape_1/assert_rank_in/static_checks_determined_all_okNoOp^while/Identity

while/img_shape_1/RankConstA^while/img_shape_1/assert_rank_in/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 

while/img_shape_1/Equal/yConstA^while/img_shape_1/assert_rank_in/static_checks_determined_all_ok*
dtype0*
_output_shapes
: *
value	B :
t
while/img_shape_1/EqualEqualwhile/img_shape_1/Rankwhile/img_shape_1/Equal/y*
T0*
_output_shapes
: 
|
while/img_shape_1/cond/SwitchSwitchwhile/img_shape_1/Equalwhile/img_shape_1/Equal*
T0
*
_output_shapes
: : 
m
while/img_shape_1/cond/switch_tIdentitywhile/img_shape_1/cond/Switch:1*
T0
*
_output_shapes
: 
k
while/img_shape_1/cond/switch_fIdentitywhile/img_shape_1/cond/Switch*
T0
*
_output_shapes
: 
d
while/img_shape_1/cond/pred_idIdentitywhile/img_shape_1/Equal*
_output_shapes
: *
T0

Д
while/img_shape_1/cond/ShapeShape%while/img_shape_1/cond/Shape/Switch:1A^while/img_shape_1/assert_rank_in/static_checks_determined_all_ok*
_output_shapes
:*
T0
љ
#while/img_shape_1/cond/Shape/SwitchSwitchwhile/TensorArrayReadV3while/img_shape_1/cond/pred_id*
T0**
_class 
loc:@while/TensorArrayReadV3*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ
й
*while/img_shape_1/cond/strided_slice/stackConstA^while/img_shape_1/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_1/cond/switch_t*
_output_shapes
:*
valueB: *
dtype0
л
,while/img_shape_1/cond/strided_slice/stack_1ConstA^while/img_shape_1/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_1/cond/switch_t*
valueB:*
dtype0*
_output_shapes
:
л
,while/img_shape_1/cond/strided_slice/stack_2ConstA^while/img_shape_1/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_1/cond/switch_t*
valueB:*
dtype0*
_output_shapes
:

$while/img_shape_1/cond/strided_sliceStridedSlicewhile/img_shape_1/cond/Shape*while/img_shape_1/cond/strided_slice/stack,while/img_shape_1/cond/strided_slice/stack_1,while/img_shape_1/cond/strided_slice/stack_2*
T0*
Index0*

begin_mask*
_output_shapes
:
Ж
while/img_shape_1/cond/Shape_1Shape%while/img_shape_1/cond/Shape_1/SwitchA^while/img_shape_1/assert_rank_in/static_checks_determined_all_ok*
_output_shapes
:*
T0
ћ
%while/img_shape_1/cond/Shape_1/SwitchSwitchwhile/TensorArrayReadV3while/img_shape_1/cond/pred_id*
T0**
_class 
loc:@while/TensorArrayReadV3*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ
л
,while/img_shape_1/cond/strided_slice_1/stackConstA^while/img_shape_1/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_1/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
н
.while/img_shape_1/cond/strided_slice_1/stack_1ConstA^while/img_shape_1/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_1/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
н
.while/img_shape_1/cond/strided_slice_1/stack_2ConstA^while/img_shape_1/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_1/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:

&while/img_shape_1/cond/strided_slice_1StridedSlicewhile/img_shape_1/cond/Shape_1,while/img_shape_1/cond/strided_slice_1/stack.while/img_shape_1/cond/strided_slice_1/stack_1.while/img_shape_1/cond/strided_slice_1/stack_2*
_output_shapes
:*
Index0*
T0
Ѓ
while/img_shape_1/cond/MergeMerge&while/img_shape_1/cond/strided_slice_1$while/img_shape_1/cond/strided_slice*
T0*
N*
_output_shapes

:: 
t
while/strided_slice/stackConst^while/Identity*
valueB: *
dtype0*
_output_shapes
:
v
while/strided_slice/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
v
while/strided_slice/stack_2Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
м
while/strided_sliceStridedSlicewhile/img_shape_1/cond/Mergewhile/strided_slice/stackwhile/strided_slice/stack_1while/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
v
while/strided_slice_1/stackConst^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_1/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_1/stack_2Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
ф
while/strided_slice_1StridedSlicewhile/img_shape_1/cond/Mergewhile/strided_slice_1/stackwhile/strided_slice_1/stack_1while/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
d
while/truediv_1/yConst^while/Identity*
dtype0*
_output_shapes
: *
value	B :
a
while/truediv_1/CastCastwhile/strided_slice*

SrcT0*
_output_shapes
: *

DstT0
a
while/truediv_1/Cast_1Castwhile/truediv_1/y*
_output_shapes
: *

DstT0*

SrcT0
i
while/truediv_1RealDivwhile/truediv_1/Castwhile/truediv_1/Cast_1*
T0*
_output_shapes
: 
D

while/CeilCeilwhile/truediv_1*
_output_shapes
: *
T0
d
while/truediv_2/yConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
while/truediv_2/CastCastwhile/strided_slice_1*

SrcT0*
_output_shapes
: *

DstT0
a
while/truediv_2/Cast_1Castwhile/truediv_2/y*

SrcT0*
_output_shapes
: *

DstT0
i
while/truediv_2RealDivwhile/truediv_2/Castwhile/truediv_2/Cast_1*
T0*
_output_shapes
: 
F
while/Ceil_1Ceilwhile/truediv_2*
T0*
_output_shapes
: 
[
while/stackPack
while/Ceilwhile/Ceil_1*
T0*
N*
_output_shapes
:
X
while/ToInt32_1Castwhile/stack*

SrcT0*
_output_shapes
:*

DstT0
v
while/strided_slice_2/stackConst^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_2/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_2/stack_2Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
г
while/strided_slice_2StridedSlicewhile/stackwhile/strided_slice_2/stackwhile/strided_slice_2/stack_1while/strided_slice_2/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
d
while/range/startConst^while/Identity*
value	B : *
dtype0*
_output_shapes
: 
d
while/range/deltaConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
[
while/range/CastCastwhile/range/start*

SrcT0*
_output_shapes
: *

DstT0
]
while/range/Cast_1Castwhile/range/delta*
_output_shapes
: *

DstT0*

SrcT0

while/rangeRangewhile/range/Castwhile/strided_slice_2while/range/Cast_1*

Tidx0*#
_output_shapes
:џџџџџџџџџ
v
while/strided_slice_3/stackConst^while/Identity*
valueB: *
dtype0*
_output_shapes
:
x
while/strided_slice_3/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_3/stack_2Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
г
while/strided_slice_3StridedSlicewhile/stackwhile/strided_slice_3/stackwhile/strided_slice_3/stack_1while/strided_slice_3/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
f
while/range_1/startConst^while/Identity*
value	B : *
dtype0*
_output_shapes
: 
f
while/range_1/deltaConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
_
while/range_1/CastCastwhile/range_1/start*
_output_shapes
: *

DstT0*

SrcT0
a
while/range_1/Cast_1Castwhile/range_1/delta*

SrcT0*
_output_shapes
: *

DstT0

while/range_1Rangewhile/range_1/Castwhile/strided_slice_3while/range_1/Cast_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0
~
while/meshgrid/Reshape/shapeConst^while/Identity*
valueB"џџџџ   *
dtype0*
_output_shapes
:
~
while/meshgrid/ReshapeReshapewhile/rangewhile/meshgrid/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

while/meshgrid/Reshape_1/shapeConst^while/Identity*
valueB"   џџџџ*
dtype0*
_output_shapes
:

while/meshgrid/Reshape_1Reshapewhile/range_1while/meshgrid/Reshape_1/shape*
T0*'
_output_shapes
:џџџџџџџџџ
I
while/meshgrid/SizeSizewhile/range*
T0*
_output_shapes
: 
M
while/meshgrid/Size_1Sizewhile/range_1*
T0*
_output_shapes
: 

while/meshgrid/Reshape_2/shapeConst^while/Identity*
valueB"   џџџџ*
dtype0*
_output_shapes
:

while/meshgrid/Reshape_2Reshapewhile/meshgrid/Reshapewhile/meshgrid/Reshape_2/shape*
T0*'
_output_shapes
:џџџџџџџџџ

while/meshgrid/Reshape_3/shapeConst^while/Identity*
_output_shapes
:*
valueB"џџџџ   *
dtype0

while/meshgrid/Reshape_3Reshapewhile/meshgrid/Reshape_1while/meshgrid/Reshape_3/shape*'
_output_shapes
:џџџџџџџџџ*
T0
k
while/meshgrid/ones/mulMulwhile/meshgrid/Size_1while/meshgrid/Size*
T0*
_output_shapes
: 
n
while/meshgrid/ones/Less/yConst^while/Identity*
value
B :ш*
dtype0*
_output_shapes
: 
v
while/meshgrid/ones/LessLesswhile/meshgrid/ones/mulwhile/meshgrid/ones/Less/y*
T0*
_output_shapes
: 
|
while/meshgrid/ones/packedPackwhile/meshgrid/Size_1while/meshgrid/Size*
T0*
N*
_output_shapes
:
s
while/meshgrid/ones/ConstConst^while/Identity*
valueB 2      №?*
dtype0*
_output_shapes
: 

while/meshgrid/onesFillwhile/meshgrid/ones/packedwhile/meshgrid/ones/Const*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

while/meshgrid/mulMulwhile/meshgrid/Reshape_2while/meshgrid/ones*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

while/meshgrid/mul_1Mulwhile/meshgrid/Reshape_3while/meshgrid/ones*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
e
while/mul/yConst^while/Identity*
valueB 2       @*
dtype0*
_output_shapes
: 
n
	while/mulMulwhile/meshgrid/mul_1while/mul/y*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
g
while/mul_1/yConst^while/Identity*
valueB 2       @*
dtype0*
_output_shapes
: 
p
while/mul_1Mulwhile/meshgrid/mulwhile/mul_1/y*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
i

while/CastCastwhile/mul_1*

SrcT0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*

DstT0
i
while/Cast_1Cast	while/mul*

SrcT0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*

DstT0
a
while/sub/yConst^while/Identity*
dtype0*
_output_shapes
: *
valueB
 *   A
f
	while/subSubwhile/Cast_1while/sub/y*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
c
while/add_1/yConst^while/Identity*
valueB
 *   A*
dtype0*
_output_shapes
: 
j
while/add_1Addwhile/Cast_1while/add_1/y*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
c
while/sub_1/yConst^while/Identity*
dtype0*
_output_shapes
: *
valueB
 *   A
h
while/sub_1Sub
while/Castwhile/sub_1/y*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
c
while/add_2/yConst^while/Identity*
valueB
 *   A*
dtype0*
_output_shapes
: 
h
while/add_2Add
while/Castwhile/add_2/y*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Є
while/stack_1Pack	while/subwhile/sub_1while/add_1while/add_2*
N*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*
axisџџџџџџџџџ
u
while/Reshape/shapeConst^while/Identity*
valueB"џџџџ   *
dtype0*
_output_shapes
:
n
while/ReshapeReshapewhile/stack_1while/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
x
%while/img_shape_2/assert_rank_in/rankConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
z
'while/img_shape_2/assert_rank_in/rank_1Const^while/Identity*
dtype0*
_output_shapes
: *
value	B :
m
&while/img_shape_2/assert_rank_in/ShapeShapewhile/TensorArrayReadV3*
T0*
_output_shapes
:
h
Owhile/img_shape_2/assert_rank_in/assert_type/statically_determined_correct_typeNoOp^while/Identity
j
Qwhile/img_shape_2/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp^while/Identity
Y
@while/img_shape_2/assert_rank_in/static_checks_determined_all_okNoOp^while/Identity

while/img_shape_2/RankConstA^while/img_shape_2/assert_rank_in/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 

while/img_shape_2/Equal/yConstA^while/img_shape_2/assert_rank_in/static_checks_determined_all_ok*
dtype0*
_output_shapes
: *
value	B :
t
while/img_shape_2/EqualEqualwhile/img_shape_2/Rankwhile/img_shape_2/Equal/y*
T0*
_output_shapes
: 
|
while/img_shape_2/cond/SwitchSwitchwhile/img_shape_2/Equalwhile/img_shape_2/Equal*
T0
*
_output_shapes
: : 
m
while/img_shape_2/cond/switch_tIdentitywhile/img_shape_2/cond/Switch:1*
_output_shapes
: *
T0

k
while/img_shape_2/cond/switch_fIdentitywhile/img_shape_2/cond/Switch*
T0
*
_output_shapes
: 
d
while/img_shape_2/cond/pred_idIdentitywhile/img_shape_2/Equal*
T0
*
_output_shapes
: 
Д
while/img_shape_2/cond/ShapeShape%while/img_shape_2/cond/Shape/Switch:1A^while/img_shape_2/assert_rank_in/static_checks_determined_all_ok*
T0*
_output_shapes
:
љ
#while/img_shape_2/cond/Shape/SwitchSwitchwhile/TensorArrayReadV3while/img_shape_2/cond/pred_id*
T0**
_class 
loc:@while/TensorArrayReadV3*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ
й
*while/img_shape_2/cond/strided_slice/stackConstA^while/img_shape_2/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_2/cond/switch_t*
valueB: *
dtype0*
_output_shapes
:
л
,while/img_shape_2/cond/strided_slice/stack_1ConstA^while/img_shape_2/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_2/cond/switch_t*
valueB:*
dtype0*
_output_shapes
:
л
,while/img_shape_2/cond/strided_slice/stack_2ConstA^while/img_shape_2/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_2/cond/switch_t*
_output_shapes
:*
valueB:*
dtype0

$while/img_shape_2/cond/strided_sliceStridedSlicewhile/img_shape_2/cond/Shape*while/img_shape_2/cond/strided_slice/stack,while/img_shape_2/cond/strided_slice/stack_1,while/img_shape_2/cond/strided_slice/stack_2*
_output_shapes
:*
Index0*
T0*

begin_mask
Ж
while/img_shape_2/cond/Shape_1Shape%while/img_shape_2/cond/Shape_1/SwitchA^while/img_shape_2/assert_rank_in/static_checks_determined_all_ok*
T0*
_output_shapes
:
ћ
%while/img_shape_2/cond/Shape_1/SwitchSwitchwhile/TensorArrayReadV3while/img_shape_2/cond/pred_id*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ*
T0**
_class 
loc:@while/TensorArrayReadV3
л
,while/img_shape_2/cond/strided_slice_1/stackConstA^while/img_shape_2/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_2/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
н
.while/img_shape_2/cond/strided_slice_1/stack_1ConstA^while/img_shape_2/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_2/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
н
.while/img_shape_2/cond/strided_slice_1/stack_2ConstA^while/img_shape_2/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_2/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:

&while/img_shape_2/cond/strided_slice_1StridedSlicewhile/img_shape_2/cond/Shape_1,while/img_shape_2/cond/strided_slice_1/stack.while/img_shape_2/cond/strided_slice_1/stack_1.while/img_shape_2/cond/strided_slice_1/stack_2*
_output_shapes
:*
T0*
Index0
Ѓ
while/img_shape_2/cond/MergeMerge&while/img_shape_2/cond/strided_slice_1$while/img_shape_2/cond/strided_slice*
N*
_output_shapes

:: *
T0
v
while/strided_slice_4/stackConst^while/Identity*
_output_shapes
:*
valueB: *
dtype0
x
while/strided_slice_4/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_4/stack_2Const^while/Identity*
dtype0*
_output_shapes
:*
valueB:
ф
while/strided_slice_4StridedSlicewhile/img_shape_2/cond/Mergewhile/strided_slice_4/stackwhile/strided_slice_4/stack_1while/strided_slice_4/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
v
while/strided_slice_5/stackConst^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_5/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_5/stack_2Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
ф
while/strided_slice_5StridedSlicewhile/img_shape_2/cond/Mergewhile/strided_slice_5/stackwhile/strided_slice_5/stack_1while/strided_slice_5/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
d
while/truediv_3/yConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
while/truediv_3/CastCastwhile/strided_slice_4*
_output_shapes
: *

DstT0*

SrcT0
a
while/truediv_3/Cast_1Castwhile/truediv_3/y*

SrcT0*
_output_shapes
: *

DstT0
i
while/truediv_3RealDivwhile/truediv_3/Castwhile/truediv_3/Cast_1*
T0*
_output_shapes
: 
F
while/Ceil_2Ceilwhile/truediv_3*
_output_shapes
: *
T0
d
while/truediv_4/yConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
while/truediv_4/CastCastwhile/strided_slice_5*

SrcT0*
_output_shapes
: *

DstT0
a
while/truediv_4/Cast_1Castwhile/truediv_4/y*
_output_shapes
: *

DstT0*

SrcT0
i
while/truediv_4RealDivwhile/truediv_4/Castwhile/truediv_4/Cast_1*
T0*
_output_shapes
: 
F
while/Ceil_3Ceilwhile/truediv_4*
_output_shapes
: *
T0
_
while/stack_2Packwhile/Ceil_2while/Ceil_3*
T0*
N*
_output_shapes
:
Z
while/ToInt32_2Castwhile/stack_2*
_output_shapes
:*

DstT0*

SrcT0
v
while/strided_slice_6/stackConst^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_6/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_6/stack_2Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
е
while/strided_slice_6StridedSlicewhile/stack_2while/strided_slice_6/stackwhile/strided_slice_6/stack_1while/strided_slice_6/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
f
while/range_2/startConst^while/Identity*
value	B : *
dtype0*
_output_shapes
: 
f
while/range_2/deltaConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
_
while/range_2/CastCastwhile/range_2/start*

SrcT0*
_output_shapes
: *

DstT0
a
while/range_2/Cast_1Castwhile/range_2/delta*

SrcT0*
_output_shapes
: *

DstT0

while/range_2Rangewhile/range_2/Castwhile/strided_slice_6while/range_2/Cast_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0
v
while/strided_slice_7/stackConst^while/Identity*
valueB: *
dtype0*
_output_shapes
:
x
while/strided_slice_7/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_7/stack_2Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
е
while/strided_slice_7StridedSlicewhile/stack_2while/strided_slice_7/stackwhile/strided_slice_7/stack_1while/strided_slice_7/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
f
while/range_3/startConst^while/Identity*
value	B : *
dtype0*
_output_shapes
: 
f
while/range_3/deltaConst^while/Identity*
_output_shapes
: *
value	B :*
dtype0
_
while/range_3/CastCastwhile/range_3/start*
_output_shapes
: *

DstT0*

SrcT0
a
while/range_3/Cast_1Castwhile/range_3/delta*

SrcT0*
_output_shapes
: *

DstT0

while/range_3Rangewhile/range_3/Castwhile/strided_slice_7while/range_3/Cast_1*

Tidx0*#
_output_shapes
:џџџџџџџџџ

while/meshgrid_1/Reshape/shapeConst^while/Identity*
valueB"џџџџ   *
dtype0*
_output_shapes
:

while/meshgrid_1/ReshapeReshapewhile/range_2while/meshgrid_1/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0

 while/meshgrid_1/Reshape_1/shapeConst^while/Identity*
valueB"   џџџџ*
dtype0*
_output_shapes
:

while/meshgrid_1/Reshape_1Reshapewhile/range_3 while/meshgrid_1/Reshape_1/shape*
T0*'
_output_shapes
:џџџџџџџџџ
M
while/meshgrid_1/SizeSizewhile/range_2*
T0*
_output_shapes
: 
O
while/meshgrid_1/Size_1Sizewhile/range_3*
T0*
_output_shapes
: 

 while/meshgrid_1/Reshape_2/shapeConst^while/Identity*
valueB"   џџџџ*
dtype0*
_output_shapes
:

while/meshgrid_1/Reshape_2Reshapewhile/meshgrid_1/Reshape while/meshgrid_1/Reshape_2/shape*
T0*'
_output_shapes
:џџџџџџџџџ

 while/meshgrid_1/Reshape_3/shapeConst^while/Identity*
valueB"џџџџ   *
dtype0*
_output_shapes
:

while/meshgrid_1/Reshape_3Reshapewhile/meshgrid_1/Reshape_1 while/meshgrid_1/Reshape_3/shape*
T0*'
_output_shapes
:џџџџџџџџџ
q
while/meshgrid_1/ones/mulMulwhile/meshgrid_1/Size_1while/meshgrid_1/Size*
T0*
_output_shapes
: 
p
while/meshgrid_1/ones/Less/yConst^while/Identity*
value
B :ш*
dtype0*
_output_shapes
: 
|
while/meshgrid_1/ones/LessLesswhile/meshgrid_1/ones/mulwhile/meshgrid_1/ones/Less/y*
T0*
_output_shapes
: 

while/meshgrid_1/ones/packedPackwhile/meshgrid_1/Size_1while/meshgrid_1/Size*
_output_shapes
:*
T0*
N
u
while/meshgrid_1/ones/ConstConst^while/Identity*
valueB 2      №?*
dtype0*
_output_shapes
: 

while/meshgrid_1/onesFillwhile/meshgrid_1/ones/packedwhile/meshgrid_1/ones/Const*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

while/meshgrid_1/mulMulwhile/meshgrid_1/Reshape_2while/meshgrid_1/ones*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

while/meshgrid_1/mul_1Mulwhile/meshgrid_1/Reshape_3while/meshgrid_1/ones*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
g
while/mul_2/yConst^while/Identity*
_output_shapes
: *
valueB 2       @*
dtype0
t
while/mul_2Mulwhile/meshgrid_1/mul_1while/mul_2/y*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
g
while/mul_3/yConst^while/Identity*
valueB 2       @*
dtype0*
_output_shapes
: 
r
while/mul_3Mulwhile/meshgrid_1/mulwhile/mul_3/y*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
k
while/Cast_2Castwhile/mul_3*

SrcT0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*

DstT0
k
while/Cast_3Castwhile/mul_2*

SrcT0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*

DstT0
c
while/sub_2/yConst^while/Identity*
valueB
 *  A*
dtype0*
_output_shapes
: 
j
while/sub_2Subwhile/Cast_3while/sub_2/y*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
c
while/add_3/yConst^while/Identity*
valueB
 *  A*
dtype0*
_output_shapes
: 
j
while/add_3Addwhile/Cast_3while/add_3/y*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
c
while/sub_3/yConst^while/Identity*
valueB
 *  A*
dtype0*
_output_shapes
: 
j
while/sub_3Subwhile/Cast_2while/sub_3/y*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
c
while/add_4/yConst^while/Identity*
valueB
 *  A*
dtype0*
_output_shapes
: 
j
while/add_4Addwhile/Cast_2while/add_4/y*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
І
while/stack_3Packwhile/sub_2while/sub_3while/add_3while/add_4*
T0*
axisџџџџџџџџџ*
N*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
w
while/Reshape_1/shapeConst^while/Identity*
valueB"џџџџ   *
dtype0*
_output_shapes
:
r
while/Reshape_1Reshapewhile/stack_3while/Reshape_1/shape*'
_output_shapes
:џџџџџџџџџ*
T0
`
while/Equal/yConst^while/Identity*
dtype0*
_output_shapes
: *
value	B : 
T
while/EqualEqualwhile/Identitywhile/Equal/y*
_output_shapes
: *
T0
X
while/cond/SwitchSwitchwhile/Equalwhile/Equal*
_output_shapes
: : *
T0

U
while/cond/switch_tIdentitywhile/cond/Switch:1*
T0
*
_output_shapes
: 
S
while/cond/switch_fIdentitywhile/cond/Switch*
_output_shapes
: *
T0

L
while/cond/pred_idIdentitywhile/Equal*
T0
*
_output_shapes
: 
х
LPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/shapeConst*>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights*%
valueB"             *
dtype0*
_output_shapes
:
Я
JPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/minConst*>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights*
valueB
 *OSО*
dtype0*
_output_shapes
: 
Я
JPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights*
valueB
 *OS>*
dtype0*
_output_shapes
: 
Б
TPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformLPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/shape*&
_output_shapes
: *
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights*
dtype0
Ъ
JPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/subSubJPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/maxJPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights*
_output_shapes
: 
ф
JPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/mulMulTPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/RandomUniformJPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/sub*&
_output_shapes
: *
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights
ж
FPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniformAddJPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/mulJPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights*&
_output_shapes
: 
Ы
+PyramidFusedNet/fem_conv0/depthwise_weights
VariableV2*
shape: *>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights*
dtype0*&
_output_shapes
: 
Ђ
2PyramidFusedNet/fem_conv0/depthwise_weights/AssignAssign+PyramidFusedNet/fem_conv0/depthwise_weightsFPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights*&
_output_shapes
: 
к
0PyramidFusedNet/fem_conv0/depthwise_weights/readIdentity+PyramidFusedNet/fem_conv0/depthwise_weights*>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights*&
_output_shapes
: *
T0
х
LPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/shapeConst*>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights*%
valueB"      `       *
dtype0*
_output_shapes
:
Я
JPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/minConst*>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights*
valueB
 *зГ]О*
dtype0*
_output_shapes
: 
Я
JPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights*
valueB
 *зГ]>*
dtype0*
_output_shapes
: 
Б
TPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformLPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/shape*&
_output_shapes
:` *
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights*
dtype0
Ъ
JPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/subSubJPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/maxJPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights*
_output_shapes
: 
ф
JPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/mulMulTPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/RandomUniformJPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights*&
_output_shapes
:` 
ж
FPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniformAddJPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/mulJPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform/min*&
_output_shapes
:` *
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights
Ы
+PyramidFusedNet/fem_conv0/pointwise_weights
VariableV2*
shape:` *>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights*
dtype0*&
_output_shapes
:` 
Ђ
2PyramidFusedNet/fem_conv0/pointwise_weights/AssignAssign+PyramidFusedNet/fem_conv0/pointwise_weightsFPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform*>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights*&
_output_shapes
:` *
T0
к
0PyramidFusedNet/fem_conv0/pointwise_weights/readIdentity+PyramidFusedNet/fem_conv0/pointwise_weights*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights*&
_output_shapes
:` 
Њ
;while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/ShapeConst^while/cond/switch_t*
_output_shapes
:*%
valueB"             *
dtype0
Њ
Cwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
й
?while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwiseDepthwiseConv2dNativeHwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Switch:1Jwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Switch_1:1*
paddingSAME*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`

Fwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/SwitchSwitchwhile/TensorArrayReadV3while/cond/pred_id*
T0**
_class 
loc:@while/TensorArrayReadV3*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ

Ewhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/EnterEnter0PyramidFusedNet/fem_conv0/depthwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
: *#

frame_namewhile/while_context
Ъ
Hwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enter*8
_output_shapes&
$: : 
Ќ
5while/cond/PyramidFusedNet/fem_conv0/separable_conv2dConv2D?while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise>while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
њ
;while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/EnterEnter0PyramidFusedNet/fem_conv0/pointwise_weights/read*
is_constant(*
parallel_iterations*&
_output_shapes
:` *#

frame_namewhile/while_context*
T0
Њ
<while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/SwitchSwitch;while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enterwhile/cond/pred_id*
T0*N
_classD
B@loc:@while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enter*8
_output_shapes&
$:` :` 
Х
:PyramidFusedNet/fem_conv0/BatchNorm/gamma/Initializer/onesConst*<
_class2
0.loc:@PyramidFusedNet/fem_conv0/BatchNorm/gamma*
valueB *  ?*
dtype0*
_output_shapes
: 
Џ
)PyramidFusedNet/fem_conv0/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes
: *
shape: *<
_class2
0.loc:@PyramidFusedNet/fem_conv0/BatchNorm/gamma

0PyramidFusedNet/fem_conv0/BatchNorm/gamma/AssignAssign)PyramidFusedNet/fem_conv0/BatchNorm/gamma:PyramidFusedNet/fem_conv0/BatchNorm/gamma/Initializer/ones*
_output_shapes
: *
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv0/BatchNorm/gamma
Ш
.PyramidFusedNet/fem_conv0/BatchNorm/gamma/readIdentity)PyramidFusedNet/fem_conv0/BatchNorm/gamma*
_output_shapes
: *
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv0/BatchNorm/gamma
Ф
:PyramidFusedNet/fem_conv0/BatchNorm/beta/Initializer/zerosConst*;
_class1
/-loc:@PyramidFusedNet/fem_conv0/BatchNorm/beta*
valueB *    *
dtype0*
_output_shapes
: 
­
(PyramidFusedNet/fem_conv0/BatchNorm/beta
VariableV2*
shape: *;
_class1
/-loc:@PyramidFusedNet/fem_conv0/BatchNorm/beta*
dtype0*
_output_shapes
: 

/PyramidFusedNet/fem_conv0/BatchNorm/beta/AssignAssign(PyramidFusedNet/fem_conv0/BatchNorm/beta:PyramidFusedNet/fem_conv0/BatchNorm/beta/Initializer/zeros*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv0/BatchNorm/beta*
_output_shapes
: 
Х
-PyramidFusedNet/fem_conv0/BatchNorm/beta/readIdentity(PyramidFusedNet/fem_conv0/BatchNorm/beta*
_output_shapes
: *
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv0/BatchNorm/beta
в
APyramidFusedNet/fem_conv0/BatchNorm/moving_mean/Initializer/zerosConst*B
_class8
64loc:@PyramidFusedNet/fem_conv0/BatchNorm/moving_mean*
valueB *    *
dtype0*
_output_shapes
: 
Л
/PyramidFusedNet/fem_conv0/BatchNorm/moving_mean
VariableV2*
_output_shapes
: *
shape: *B
_class8
64loc:@PyramidFusedNet/fem_conv0/BatchNorm/moving_mean*
dtype0

6PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/AssignAssign/PyramidFusedNet/fem_conv0/BatchNorm/moving_meanAPyramidFusedNet/fem_conv0/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes
: *
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv0/BatchNorm/moving_mean
к
4PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/readIdentity/PyramidFusedNet/fem_conv0/BatchNorm/moving_mean*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv0/BatchNorm/moving_mean*
_output_shapes
: 
й
DPyramidFusedNet/fem_conv0/BatchNorm/moving_variance/Initializer/onesConst*F
_class<
:8loc:@PyramidFusedNet/fem_conv0/BatchNorm/moving_variance*
valueB *  ?*
dtype0*
_output_shapes
: 
У
3PyramidFusedNet/fem_conv0/BatchNorm/moving_variance
VariableV2*F
_class<
:8loc:@PyramidFusedNet/fem_conv0/BatchNorm/moving_variance*
dtype0*
_output_shapes
: *
shape: 
Ќ
:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/AssignAssign3PyramidFusedNet/fem_conv0/BatchNorm/moving_varianceDPyramidFusedNet/fem_conv0/BatchNorm/moving_variance/Initializer/ones*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv0/BatchNorm/moving_variance*
_output_shapes
: 
ц
8PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/readIdentity3PyramidFusedNet/fem_conv0/BatchNorm/moving_variance*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv0/BatchNorm/moving_variance*
_output_shapes
: 
Ќ
=while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNormFusedBatchNorm5while/cond/PyramidFusedNet/fem_conv0/separable_conv2dFwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch:1Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_1:1Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_2:1Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_3:1*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ : : : : *
is_training( *
epsilon%o:
є
Cwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/EnterEnter.PyramidFusedNet/fem_conv0/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
Њ
Dwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/SwitchSwitchCwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id* 
_output_shapes
: : *
T0*V
_classL
JHloc:@while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter
ѕ
Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1Enter-PyramidFusedNet/fem_conv0/BatchNorm/beta/read*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context*
T0
А
Fwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
: : 
ќ
Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2Enter4PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
А
Fwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_2SwitchEwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
: : 

Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3Enter8PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/read*
_output_shapes
: *#

frame_namewhile/while_context*
T0*
is_constant(*
parallel_iterations
А
Fwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_3SwitchEwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
: : 

4while/cond/PyramidFusedNet/fem_conv0/BatchNorm/ConstConst^while/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *wО?

4while/cond/PyramidFusedNet/fem_conv0/LeakyRelu/alphaConst^while/cond/switch_t*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ё
2while/cond/PyramidFusedNet/fem_conv0/LeakyRelu/mulMul4while/cond/PyramidFusedNet/fem_conv0/LeakyRelu/alpha=while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
я
.while/cond/PyramidFusedNet/fem_conv0/LeakyReluMaximum2while/cond/PyramidFusedNet/fem_conv0/LeakyRelu/mul=while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
х
LPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/shapeConst*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights*%
valueB"             *
dtype0*
_output_shapes
:
Я
JPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/minConst*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights*
valueB
 *OSО*
dtype0*
_output_shapes
: 
Я
JPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights*
valueB
 *OS>*
dtype0*
_output_shapes
: 
Б
TPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformLPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights*
dtype0*&
_output_shapes
: 
Ъ
JPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/subSubJPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/maxJPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights*
_output_shapes
: 
ф
JPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/mulMulTPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/RandomUniformJPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/sub*&
_output_shapes
: *
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights
ж
FPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniformAddJPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/mulJPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights*&
_output_shapes
: 
Ы
+PyramidFusedNet/fem_conv1/depthwise_weights
VariableV2*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights*
dtype0*&
_output_shapes
: *
shape: 
Ђ
2PyramidFusedNet/fem_conv1/depthwise_weights/AssignAssign+PyramidFusedNet/fem_conv1/depthwise_weightsFPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights*&
_output_shapes
: 
к
0PyramidFusedNet/fem_conv1/depthwise_weights/readIdentity+PyramidFusedNet/fem_conv1/depthwise_weights*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights*&
_output_shapes
: 
х
LPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*%
valueB"      `   @   
Я
JPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/minConst*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*
valueB
 *јKFО*
dtype0*
_output_shapes
: 
Я
JPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*
valueB
 *јKF>*
dtype0*
_output_shapes
: 
Б
TPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformLPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*
dtype0*&
_output_shapes
:`@
Ъ
JPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/subSubJPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/maxJPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*
_output_shapes
: 
ф
JPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/mulMulTPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/RandomUniformJPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*&
_output_shapes
:`@
ж
FPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniformAddJPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/mulJPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*&
_output_shapes
:`@
Ы
+PyramidFusedNet/fem_conv1/pointwise_weights
VariableV2*
shape:`@*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*
dtype0*&
_output_shapes
:`@
Ђ
2PyramidFusedNet/fem_conv1/pointwise_weights/AssignAssign+PyramidFusedNet/fem_conv1/pointwise_weightsFPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*&
_output_shapes
:`@
к
0PyramidFusedNet/fem_conv1/pointwise_weights/readIdentity+PyramidFusedNet/fem_conv1/pointwise_weights*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*&
_output_shapes
:`@
Њ
;while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"             *
dtype0*
_output_shapes
:
Њ
Cwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/dilation_rateConst^while/cond/switch_t*
_output_shapes
:*
valueB"      *
dtype0
Н
?while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwiseDepthwiseConv2dNative.while/cond/PyramidFusedNet/fem_conv0/LeakyReluHwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Switch:1*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`*
paddingSAME*
T0

Ewhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/EnterEnter0PyramidFusedNet/fem_conv1/depthwise_weights/read*
parallel_iterations*&
_output_shapes
: *#

frame_namewhile/while_context*
T0*
is_constant(
Ш
Fwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/SwitchSwitchEwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enterwhile/cond/pred_id*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enter*8
_output_shapes&
$: : *
T0
Ќ
5while/cond/PyramidFusedNet/fem_conv1/separable_conv2dConv2D?while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise>while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
њ
;while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/EnterEnter0PyramidFusedNet/fem_conv1/pointwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
:`@*#

frame_namewhile/while_context
Њ
<while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/SwitchSwitch;while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enterwhile/cond/pred_id*
T0*N
_classD
B@loc:@while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enter*8
_output_shapes&
$:`@:`@
Х
:PyramidFusedNet/fem_conv1/BatchNorm/gamma/Initializer/onesConst*<
_class2
0.loc:@PyramidFusedNet/fem_conv1/BatchNorm/gamma*
valueB@*  ?*
dtype0*
_output_shapes
:@
Џ
)PyramidFusedNet/fem_conv1/BatchNorm/gamma
VariableV2*<
_class2
0.loc:@PyramidFusedNet/fem_conv1/BatchNorm/gamma*
dtype0*
_output_shapes
:@*
shape:@

0PyramidFusedNet/fem_conv1/BatchNorm/gamma/AssignAssign)PyramidFusedNet/fem_conv1/BatchNorm/gamma:PyramidFusedNet/fem_conv1/BatchNorm/gamma/Initializer/ones*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv1/BatchNorm/gamma*
_output_shapes
:@
Ш
.PyramidFusedNet/fem_conv1/BatchNorm/gamma/readIdentity)PyramidFusedNet/fem_conv1/BatchNorm/gamma*
_output_shapes
:@*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv1/BatchNorm/gamma
Ф
:PyramidFusedNet/fem_conv1/BatchNorm/beta/Initializer/zerosConst*;
_class1
/-loc:@PyramidFusedNet/fem_conv1/BatchNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
­
(PyramidFusedNet/fem_conv1/BatchNorm/beta
VariableV2*;
_class1
/-loc:@PyramidFusedNet/fem_conv1/BatchNorm/beta*
dtype0*
_output_shapes
:@*
shape:@

/PyramidFusedNet/fem_conv1/BatchNorm/beta/AssignAssign(PyramidFusedNet/fem_conv1/BatchNorm/beta:PyramidFusedNet/fem_conv1/BatchNorm/beta/Initializer/zeros*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv1/BatchNorm/beta*
_output_shapes
:@
Х
-PyramidFusedNet/fem_conv1/BatchNorm/beta/readIdentity(PyramidFusedNet/fem_conv1/BatchNorm/beta*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv1/BatchNorm/beta*
_output_shapes
:@
в
APyramidFusedNet/fem_conv1/BatchNorm/moving_mean/Initializer/zerosConst*B
_class8
64loc:@PyramidFusedNet/fem_conv1/BatchNorm/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
Л
/PyramidFusedNet/fem_conv1/BatchNorm/moving_mean
VariableV2*
shape:@*B
_class8
64loc:@PyramidFusedNet/fem_conv1/BatchNorm/moving_mean*
dtype0*
_output_shapes
:@

6PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/AssignAssign/PyramidFusedNet/fem_conv1/BatchNorm/moving_meanAPyramidFusedNet/fem_conv1/BatchNorm/moving_mean/Initializer/zeros*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv1/BatchNorm/moving_mean*
_output_shapes
:@
к
4PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/readIdentity/PyramidFusedNet/fem_conv1/BatchNorm/moving_mean*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv1/BatchNorm/moving_mean*
_output_shapes
:@
й
DPyramidFusedNet/fem_conv1/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes
:@*F
_class<
:8loc:@PyramidFusedNet/fem_conv1/BatchNorm/moving_variance*
valueB@*  ?*
dtype0
У
3PyramidFusedNet/fem_conv1/BatchNorm/moving_variance
VariableV2*
shape:@*F
_class<
:8loc:@PyramidFusedNet/fem_conv1/BatchNorm/moving_variance*
dtype0*
_output_shapes
:@
Ќ
:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/AssignAssign3PyramidFusedNet/fem_conv1/BatchNorm/moving_varianceDPyramidFusedNet/fem_conv1/BatchNorm/moving_variance/Initializer/ones*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv1/BatchNorm/moving_variance*
_output_shapes
:@
ц
8PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/readIdentity3PyramidFusedNet/fem_conv1/BatchNorm/moving_variance*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv1/BatchNorm/moving_variance*
_output_shapes
:@
Ќ
=while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNormFusedBatchNorm5while/cond/PyramidFusedNet/fem_conv1/separable_conv2dFwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch:1Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_1:1Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_2:1Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_3:1*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( 
є
Cwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/EnterEnter.PyramidFusedNet/fem_conv1/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
Њ
Dwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/SwitchSwitchCwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*V
_classL
JHloc:@while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
:@:@
ѕ
Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1Enter-PyramidFusedNet/fem_conv1/BatchNorm/beta/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
А
Fwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id* 
_output_shapes
:@:@*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1
ќ
Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2Enter4PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
А
Fwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_2SwitchEwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
:@:@

Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3Enter8PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/read*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context*
T0*
is_constant(
А
Fwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_3SwitchEwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
:@:@*
T0

4while/cond/PyramidFusedNet/fem_conv1/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

4while/cond/PyramidFusedNet/fem_conv1/LeakyRelu/alphaConst^while/cond/switch_t*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ё
2while/cond/PyramidFusedNet/fem_conv1/LeakyRelu/mulMul4while/cond/PyramidFusedNet/fem_conv1/LeakyRelu/alpha=while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
я
.while/cond/PyramidFusedNet/fem_conv1/LeakyReluMaximum2while/cond/PyramidFusedNet/fem_conv1/LeakyRelu/mul=while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
х
LPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/shapeConst*>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights*%
valueB"      @      *
dtype0*
_output_shapes
:
Я
JPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/minConst*
_output_shapes
: *>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights*
valueB
 *8JЬН*
dtype0
Я
JPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights*
valueB
 *8JЬ=*
dtype0*
_output_shapes
: 
Б
TPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformLPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights
Ъ
JPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/subSubJPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/maxJPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights*
_output_shapes
: 
ф
JPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/mulMulTPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/RandomUniformJPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights*&
_output_shapes
:@
ж
FPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniformAddJPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/mulJPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform/min*&
_output_shapes
:@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights
Ы
+PyramidFusedNet/fem_conv2/depthwise_weights
VariableV2*>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights*
dtype0*&
_output_shapes
:@*
shape:@
Ђ
2PyramidFusedNet/fem_conv2/depthwise_weights/AssignAssign+PyramidFusedNet/fem_conv2/depthwise_weightsFPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights*&
_output_shapes
:@
к
0PyramidFusedNet/fem_conv2/depthwise_weights/readIdentity+PyramidFusedNet/fem_conv2/depthwise_weights*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights*&
_output_shapes
:@
х
LPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/shapeConst*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights*%
valueB"      Р   @   *
dtype0*
_output_shapes
:
Я
JPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/minConst*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights*
valueB
 *qФО*
dtype0*
_output_shapes
: 
Я
JPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights*
valueB
 *qФ>*
dtype0*
_output_shapes
: 
В
TPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformLPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights*
dtype0*'
_output_shapes
:Р@
Ъ
JPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/subSubJPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/maxJPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights*
_output_shapes
: 
х
JPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/mulMulTPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/RandomUniformJPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/sub*'
_output_shapes
:Р@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights
з
FPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniformAddJPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/mulJPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform/min*'
_output_shapes
:Р@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights
Э
+PyramidFusedNet/fem_conv2/pointwise_weights
VariableV2*'
_output_shapes
:Р@*
shape:Р@*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights*
dtype0
Ѓ
2PyramidFusedNet/fem_conv2/pointwise_weights/AssignAssign+PyramidFusedNet/fem_conv2/pointwise_weightsFPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform*'
_output_shapes
:Р@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights
л
0PyramidFusedNet/fem_conv2/pointwise_weights/readIdentity+PyramidFusedNet/fem_conv2/pointwise_weights*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights*'
_output_shapes
:Р@
Њ
;while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"      @      *
dtype0*
_output_shapes
:
Њ
Cwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
О
?while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwiseDepthwiseConv2dNative.while/cond/PyramidFusedNet/fem_conv1/LeakyReluHwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР

Ewhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/EnterEnter0PyramidFusedNet/fem_conv2/depthwise_weights/read*
is_constant(*
parallel_iterations*&
_output_shapes
:@*#

frame_namewhile/while_context*
T0
Ш
Fwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/SwitchSwitchEwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enter*8
_output_shapes&
$:@:@
Ќ
5while/cond/PyramidFusedNet/fem_conv2/separable_conv2dConv2D?while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise>while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Switch:1*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
paddingVALID*
T0*
strides

ћ
;while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/EnterEnter0PyramidFusedNet/fem_conv2/pointwise_weights/read*
parallel_iterations*'
_output_shapes
:Р@*#

frame_namewhile/while_context*
T0*
is_constant(
Ќ
<while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/SwitchSwitch;while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enterwhile/cond/pred_id*
T0*N
_classD
B@loc:@while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enter*:
_output_shapes(
&:Р@:Р@
Х
:PyramidFusedNet/fem_conv2/BatchNorm/gamma/Initializer/onesConst*<
_class2
0.loc:@PyramidFusedNet/fem_conv2/BatchNorm/gamma*
valueB@*  ?*
dtype0*
_output_shapes
:@
Џ
)PyramidFusedNet/fem_conv2/BatchNorm/gamma
VariableV2*<
_class2
0.loc:@PyramidFusedNet/fem_conv2/BatchNorm/gamma*
dtype0*
_output_shapes
:@*
shape:@

0PyramidFusedNet/fem_conv2/BatchNorm/gamma/AssignAssign)PyramidFusedNet/fem_conv2/BatchNorm/gamma:PyramidFusedNet/fem_conv2/BatchNorm/gamma/Initializer/ones*
_output_shapes
:@*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv2/BatchNorm/gamma
Ш
.PyramidFusedNet/fem_conv2/BatchNorm/gamma/readIdentity)PyramidFusedNet/fem_conv2/BatchNorm/gamma*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv2/BatchNorm/gamma*
_output_shapes
:@
Ф
:PyramidFusedNet/fem_conv2/BatchNorm/beta/Initializer/zerosConst*;
_class1
/-loc:@PyramidFusedNet/fem_conv2/BatchNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
­
(PyramidFusedNet/fem_conv2/BatchNorm/beta
VariableV2*;
_class1
/-loc:@PyramidFusedNet/fem_conv2/BatchNorm/beta*
dtype0*
_output_shapes
:@*
shape:@

/PyramidFusedNet/fem_conv2/BatchNorm/beta/AssignAssign(PyramidFusedNet/fem_conv2/BatchNorm/beta:PyramidFusedNet/fem_conv2/BatchNorm/beta/Initializer/zeros*;
_class1
/-loc:@PyramidFusedNet/fem_conv2/BatchNorm/beta*
_output_shapes
:@*
T0
Х
-PyramidFusedNet/fem_conv2/BatchNorm/beta/readIdentity(PyramidFusedNet/fem_conv2/BatchNorm/beta*
_output_shapes
:@*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv2/BatchNorm/beta
в
APyramidFusedNet/fem_conv2/BatchNorm/moving_mean/Initializer/zerosConst*B
_class8
64loc:@PyramidFusedNet/fem_conv2/BatchNorm/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
Л
/PyramidFusedNet/fem_conv2/BatchNorm/moving_mean
VariableV2*B
_class8
64loc:@PyramidFusedNet/fem_conv2/BatchNorm/moving_mean*
dtype0*
_output_shapes
:@*
shape:@

6PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/AssignAssign/PyramidFusedNet/fem_conv2/BatchNorm/moving_meanAPyramidFusedNet/fem_conv2/BatchNorm/moving_mean/Initializer/zeros*B
_class8
64loc:@PyramidFusedNet/fem_conv2/BatchNorm/moving_mean*
_output_shapes
:@*
T0
к
4PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/readIdentity/PyramidFusedNet/fem_conv2/BatchNorm/moving_mean*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv2/BatchNorm/moving_mean*
_output_shapes
:@
й
DPyramidFusedNet/fem_conv2/BatchNorm/moving_variance/Initializer/onesConst*F
_class<
:8loc:@PyramidFusedNet/fem_conv2/BatchNorm/moving_variance*
valueB@*  ?*
dtype0*
_output_shapes
:@
У
3PyramidFusedNet/fem_conv2/BatchNorm/moving_variance
VariableV2*F
_class<
:8loc:@PyramidFusedNet/fem_conv2/BatchNorm/moving_variance*
dtype0*
_output_shapes
:@*
shape:@
Ќ
:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/AssignAssign3PyramidFusedNet/fem_conv2/BatchNorm/moving_varianceDPyramidFusedNet/fem_conv2/BatchNorm/moving_variance/Initializer/ones*F
_class<
:8loc:@PyramidFusedNet/fem_conv2/BatchNorm/moving_variance*
_output_shapes
:@*
T0
ц
8PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/readIdentity3PyramidFusedNet/fem_conv2/BatchNorm/moving_variance*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv2/BatchNorm/moving_variance*
_output_shapes
:@
Ќ
=while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNormFusedBatchNorm5while/cond/PyramidFusedNet/fem_conv2/separable_conv2dFwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch:1Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_1:1Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_2:1Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_3:1*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( *
epsilon%o:
є
Cwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/EnterEnter.PyramidFusedNet/fem_conv2/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
Њ
Dwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/SwitchSwitchCwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*V
_classL
JHloc:@while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
:@:@
ѕ
Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1Enter-PyramidFusedNet/fem_conv2/BatchNorm/beta/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
А
Fwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
:@:@
ќ
Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2Enter4PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
А
Fwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_2SwitchEwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id* 
_output_shapes
:@:@*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2

Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3Enter8PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/read*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context*
T0
А
Fwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_3SwitchEwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
:@:@

4while/cond/PyramidFusedNet/fem_conv2/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

4while/cond/PyramidFusedNet/fem_conv2/LeakyRelu/alphaConst^while/cond/switch_t*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ё
2while/cond/PyramidFusedNet/fem_conv2/LeakyRelu/mulMul4while/cond/PyramidFusedNet/fem_conv2/LeakyRelu/alpha=while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
я
.while/cond/PyramidFusedNet/fem_conv2/LeakyReluMaximum2while/cond/PyramidFusedNet/fem_conv2/LeakyRelu/mul=while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
х
LPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/shapeConst*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights*%
valueB"      @      *
dtype0*
_output_shapes
:
Я
JPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/minConst*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights*
valueB
 *8JЬН*
dtype0*
_output_shapes
: 
Я
JPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights*
valueB
 *8JЬ=*
dtype0*
_output_shapes
: 
Б
TPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformLPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/shape*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights*
dtype0*&
_output_shapes
:@*
T0
Ъ
JPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/subSubJPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/maxJPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights*
_output_shapes
: 
ф
JPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/mulMulTPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/RandomUniformJPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/sub*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights*&
_output_shapes
:@*
T0
ж
FPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniformAddJPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/mulJPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights*&
_output_shapes
:@
Ы
+PyramidFusedNet/fem_conv3/depthwise_weights
VariableV2*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights*
dtype0*&
_output_shapes
:@*
shape:@
Ђ
2PyramidFusedNet/fem_conv3/depthwise_weights/AssignAssign+PyramidFusedNet/fem_conv3/depthwise_weightsFPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform*&
_output_shapes
:@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights
к
0PyramidFusedNet/fem_conv3/depthwise_weights/readIdentity+PyramidFusedNet/fem_conv3/depthwise_weights*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights*&
_output_shapes
:@
х
LPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/shapeConst*>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights*%
valueB"      Р   @   *
dtype0*
_output_shapes
:
Я
JPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/minConst*
_output_shapes
: *>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights*
valueB
 *qФО*
dtype0
Я
JPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights*
valueB
 *qФ>
В
TPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformLPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:Р@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights
Ъ
JPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/subSubJPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/maxJPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights*
_output_shapes
: 
х
JPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/mulMulTPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/RandomUniformJPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/sub*'
_output_shapes
:Р@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights
з
FPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniformAddJPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/mulJPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights*'
_output_shapes
:Р@
Э
+PyramidFusedNet/fem_conv3/pointwise_weights
VariableV2*
shape:Р@*>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights*
dtype0*'
_output_shapes
:Р@
Ѓ
2PyramidFusedNet/fem_conv3/pointwise_weights/AssignAssign+PyramidFusedNet/fem_conv3/pointwise_weightsFPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights*'
_output_shapes
:Р@
л
0PyramidFusedNet/fem_conv3/pointwise_weights/readIdentity+PyramidFusedNet/fem_conv3/pointwise_weights*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights*'
_output_shapes
:Р@
Њ
;while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"      @      *
dtype0*
_output_shapes
:
Њ
Cwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/dilation_rateConst^while/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"      
О
?while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwiseDepthwiseConv2dNative.while/cond/PyramidFusedNet/fem_conv2/LeakyReluHwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР

Ewhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/EnterEnter0PyramidFusedNet/fem_conv3/depthwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
:@*#

frame_namewhile/while_context
Ш
Fwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/SwitchSwitchEwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enter*8
_output_shapes&
$:@:@
Ќ
5while/cond/PyramidFusedNet/fem_conv3/separable_conv2dConv2D?while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise>while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Switch:1*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
paddingVALID*
T0*
strides

ћ
;while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/EnterEnter0PyramidFusedNet/fem_conv3/pointwise_weights/read*
parallel_iterations*'
_output_shapes
:Р@*#

frame_namewhile/while_context*
T0*
is_constant(
Ќ
<while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/SwitchSwitch;while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enterwhile/cond/pred_id*
T0*N
_classD
B@loc:@while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enter*:
_output_shapes(
&:Р@:Р@
Х
:PyramidFusedNet/fem_conv3/BatchNorm/gamma/Initializer/onesConst*<
_class2
0.loc:@PyramidFusedNet/fem_conv3/BatchNorm/gamma*
valueB@*  ?*
dtype0*
_output_shapes
:@
Џ
)PyramidFusedNet/fem_conv3/BatchNorm/gamma
VariableV2*
shape:@*<
_class2
0.loc:@PyramidFusedNet/fem_conv3/BatchNorm/gamma*
dtype0*
_output_shapes
:@

0PyramidFusedNet/fem_conv3/BatchNorm/gamma/AssignAssign)PyramidFusedNet/fem_conv3/BatchNorm/gamma:PyramidFusedNet/fem_conv3/BatchNorm/gamma/Initializer/ones*
_output_shapes
:@*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv3/BatchNorm/gamma
Ш
.PyramidFusedNet/fem_conv3/BatchNorm/gamma/readIdentity)PyramidFusedNet/fem_conv3/BatchNorm/gamma*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv3/BatchNorm/gamma*
_output_shapes
:@
Ф
:PyramidFusedNet/fem_conv3/BatchNorm/beta/Initializer/zerosConst*;
_class1
/-loc:@PyramidFusedNet/fem_conv3/BatchNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
­
(PyramidFusedNet/fem_conv3/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes
:@*
shape:@*;
_class1
/-loc:@PyramidFusedNet/fem_conv3/BatchNorm/beta

/PyramidFusedNet/fem_conv3/BatchNorm/beta/AssignAssign(PyramidFusedNet/fem_conv3/BatchNorm/beta:PyramidFusedNet/fem_conv3/BatchNorm/beta/Initializer/zeros*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv3/BatchNorm/beta*
_output_shapes
:@
Х
-PyramidFusedNet/fem_conv3/BatchNorm/beta/readIdentity(PyramidFusedNet/fem_conv3/BatchNorm/beta*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv3/BatchNorm/beta*
_output_shapes
:@
в
APyramidFusedNet/fem_conv3/BatchNorm/moving_mean/Initializer/zerosConst*B
_class8
64loc:@PyramidFusedNet/fem_conv3/BatchNorm/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
Л
/PyramidFusedNet/fem_conv3/BatchNorm/moving_mean
VariableV2*B
_class8
64loc:@PyramidFusedNet/fem_conv3/BatchNorm/moving_mean*
dtype0*
_output_shapes
:@*
shape:@

6PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/AssignAssign/PyramidFusedNet/fem_conv3/BatchNorm/moving_meanAPyramidFusedNet/fem_conv3/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes
:@*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv3/BatchNorm/moving_mean
к
4PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/readIdentity/PyramidFusedNet/fem_conv3/BatchNorm/moving_mean*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv3/BatchNorm/moving_mean*
_output_shapes
:@
й
DPyramidFusedNet/fem_conv3/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:@*F
_class<
:8loc:@PyramidFusedNet/fem_conv3/BatchNorm/moving_variance*
valueB@*  ?
У
3PyramidFusedNet/fem_conv3/BatchNorm/moving_variance
VariableV2*
shape:@*F
_class<
:8loc:@PyramidFusedNet/fem_conv3/BatchNorm/moving_variance*
dtype0*
_output_shapes
:@
Ќ
:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/AssignAssign3PyramidFusedNet/fem_conv3/BatchNorm/moving_varianceDPyramidFusedNet/fem_conv3/BatchNorm/moving_variance/Initializer/ones*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv3/BatchNorm/moving_variance*
_output_shapes
:@
ц
8PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/readIdentity3PyramidFusedNet/fem_conv3/BatchNorm/moving_variance*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv3/BatchNorm/moving_variance*
_output_shapes
:@
Ќ
=while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNormFusedBatchNorm5while/cond/PyramidFusedNet/fem_conv3/separable_conv2dFwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch:1Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_1:1Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_2:1Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_3:1*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( *
epsilon%o:*
T0
є
Cwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/EnterEnter.PyramidFusedNet/fem_conv3/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
Њ
Dwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/SwitchSwitchCwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*V
_classL
JHloc:@while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
:@:@
ѕ
Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1Enter-PyramidFusedNet/fem_conv3/BatchNorm/beta/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
А
Fwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id* 
_output_shapes
:@:@*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1
ќ
Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2Enter4PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/read*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context*
T0*
is_constant(
А
Fwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_2SwitchEwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id* 
_output_shapes
:@:@*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2

Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3Enter8PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
А
Fwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_3SwitchEwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
:@:@*
T0

4while/cond/PyramidFusedNet/fem_conv3/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

4while/cond/PyramidFusedNet/fem_conv3/LeakyRelu/alphaConst^while/cond/switch_t*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ё
2while/cond/PyramidFusedNet/fem_conv3/LeakyRelu/mulMul4while/cond/PyramidFusedNet/fem_conv3/LeakyRelu/alpha=while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
я
.while/cond/PyramidFusedNet/fem_conv3/LeakyReluMaximum2while/cond/PyramidFusedNet/fem_conv3/LeakyRelu/mul=while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
х
LPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights*%
valueB"      @      *
dtype0
Я
JPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/minConst*
_output_shapes
: *>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights*
valueB
 *8JЬН*
dtype0
Я
JPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/maxConst*
_output_shapes
: *>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights*
valueB
 *8JЬ=*
dtype0
Б
TPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformLPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights*
dtype0*&
_output_shapes
:@
Ъ
JPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/subSubJPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/maxJPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights
ф
JPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/mulMulTPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/RandomUniformJPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/sub*&
_output_shapes
:@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights
ж
FPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniformAddJPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/mulJPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform/min*>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights*&
_output_shapes
:@*
T0
Ы
+PyramidFusedNet/fem_conv4/depthwise_weights
VariableV2*>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights*
dtype0*&
_output_shapes
:@*
shape:@
Ђ
2PyramidFusedNet/fem_conv4/depthwise_weights/AssignAssign+PyramidFusedNet/fem_conv4/depthwise_weightsFPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights*&
_output_shapes
:@
к
0PyramidFusedNet/fem_conv4/depthwise_weights/readIdentity+PyramidFusedNet/fem_conv4/depthwise_weights*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights*&
_output_shapes
:@
х
LPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights*%
valueB"      Р   @   
Я
JPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/minConst*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights*
valueB
 *qФО*
dtype0*
_output_shapes
: 
Я
JPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/maxConst*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights*
valueB
 *qФ>*
dtype0*
_output_shapes
: 
В
TPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformLPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights*
dtype0*'
_output_shapes
:Р@
Ъ
JPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/subSubJPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/maxJPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/min*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights*
_output_shapes
: 
х
JPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/mulMulTPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/RandomUniformJPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights*'
_output_shapes
:Р@
з
FPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniformAddJPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/mulJPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform/min*'
_output_shapes
:Р@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights
Э
+PyramidFusedNet/fem_conv4/pointwise_weights
VariableV2*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights*
dtype0*'
_output_shapes
:Р@*
shape:Р@
Ѓ
2PyramidFusedNet/fem_conv4/pointwise_weights/AssignAssign+PyramidFusedNet/fem_conv4/pointwise_weightsFPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform*'
_output_shapes
:Р@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights
л
0PyramidFusedNet/fem_conv4/pointwise_weights/readIdentity+PyramidFusedNet/fem_conv4/pointwise_weights*'
_output_shapes
:Р@*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights
Њ
;while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"      @      *
dtype0*
_output_shapes
:
Њ
Cwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
О
?while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwiseDepthwiseConv2dNative.while/cond/PyramidFusedNet/fem_conv3/LeakyReluHwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР

Ewhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/EnterEnter0PyramidFusedNet/fem_conv4/depthwise_weights/read*&
_output_shapes
:@*#

frame_namewhile/while_context*
T0*
is_constant(*
parallel_iterations
Ш
Fwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/SwitchSwitchEwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enterwhile/cond/pred_id*8
_output_shapes&
$:@:@*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enter
Ќ
5while/cond/PyramidFusedNet/fem_conv4/separable_conv2dConv2D?while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise>while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
ћ
;while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/EnterEnter0PyramidFusedNet/fem_conv4/pointwise_weights/read*
T0*
is_constant(*
parallel_iterations*'
_output_shapes
:Р@*#

frame_namewhile/while_context
Ќ
<while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/SwitchSwitch;while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enterwhile/cond/pred_id*:
_output_shapes(
&:Р@:Р@*
T0*N
_classD
B@loc:@while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enter
Х
:PyramidFusedNet/fem_conv4/BatchNorm/gamma/Initializer/onesConst*<
_class2
0.loc:@PyramidFusedNet/fem_conv4/BatchNorm/gamma*
valueB@*  ?*
dtype0*
_output_shapes
:@
Џ
)PyramidFusedNet/fem_conv4/BatchNorm/gamma
VariableV2*<
_class2
0.loc:@PyramidFusedNet/fem_conv4/BatchNorm/gamma*
dtype0*
_output_shapes
:@*
shape:@

0PyramidFusedNet/fem_conv4/BatchNorm/gamma/AssignAssign)PyramidFusedNet/fem_conv4/BatchNorm/gamma:PyramidFusedNet/fem_conv4/BatchNorm/gamma/Initializer/ones*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv4/BatchNorm/gamma*
_output_shapes
:@
Ш
.PyramidFusedNet/fem_conv4/BatchNorm/gamma/readIdentity)PyramidFusedNet/fem_conv4/BatchNorm/gamma*
_output_shapes
:@*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv4/BatchNorm/gamma
Ф
:PyramidFusedNet/fem_conv4/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*;
_class1
/-loc:@PyramidFusedNet/fem_conv4/BatchNorm/beta*
valueB@*    
­
(PyramidFusedNet/fem_conv4/BatchNorm/beta
VariableV2*;
_class1
/-loc:@PyramidFusedNet/fem_conv4/BatchNorm/beta*
dtype0*
_output_shapes
:@*
shape:@

/PyramidFusedNet/fem_conv4/BatchNorm/beta/AssignAssign(PyramidFusedNet/fem_conv4/BatchNorm/beta:PyramidFusedNet/fem_conv4/BatchNorm/beta/Initializer/zeros*
_output_shapes
:@*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv4/BatchNorm/beta
Х
-PyramidFusedNet/fem_conv4/BatchNorm/beta/readIdentity(PyramidFusedNet/fem_conv4/BatchNorm/beta*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv4/BatchNorm/beta*
_output_shapes
:@
в
APyramidFusedNet/fem_conv4/BatchNorm/moving_mean/Initializer/zerosConst*B
_class8
64loc:@PyramidFusedNet/fem_conv4/BatchNorm/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
Л
/PyramidFusedNet/fem_conv4/BatchNorm/moving_mean
VariableV2*B
_class8
64loc:@PyramidFusedNet/fem_conv4/BatchNorm/moving_mean*
dtype0*
_output_shapes
:@*
shape:@

6PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/AssignAssign/PyramidFusedNet/fem_conv4/BatchNorm/moving_meanAPyramidFusedNet/fem_conv4/BatchNorm/moving_mean/Initializer/zeros*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv4/BatchNorm/moving_mean*
_output_shapes
:@
к
4PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/readIdentity/PyramidFusedNet/fem_conv4/BatchNorm/moving_mean*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv4/BatchNorm/moving_mean*
_output_shapes
:@
й
DPyramidFusedNet/fem_conv4/BatchNorm/moving_variance/Initializer/onesConst*F
_class<
:8loc:@PyramidFusedNet/fem_conv4/BatchNorm/moving_variance*
valueB@*  ?*
dtype0*
_output_shapes
:@
У
3PyramidFusedNet/fem_conv4/BatchNorm/moving_variance
VariableV2*
shape:@*F
_class<
:8loc:@PyramidFusedNet/fem_conv4/BatchNorm/moving_variance*
dtype0*
_output_shapes
:@
Ќ
:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/AssignAssign3PyramidFusedNet/fem_conv4/BatchNorm/moving_varianceDPyramidFusedNet/fem_conv4/BatchNorm/moving_variance/Initializer/ones*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv4/BatchNorm/moving_variance*
_output_shapes
:@
ц
8PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/readIdentity3PyramidFusedNet/fem_conv4/BatchNorm/moving_variance*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv4/BatchNorm/moving_variance*
_output_shapes
:@
Ќ
=while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNormFusedBatchNorm5while/cond/PyramidFusedNet/fem_conv4/separable_conv2dFwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch:1Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_1:1Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_2:1Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_3:1*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( *
epsilon%o:*
T0
є
Cwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/EnterEnter.PyramidFusedNet/fem_conv4/BatchNorm/gamma/read*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context*
T0*
is_constant(
Њ
Dwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/SwitchSwitchCwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*V
_classL
JHloc:@while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
:@:@
ѕ
Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1Enter-PyramidFusedNet/fem_conv4/BatchNorm/beta/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
А
Fwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id* 
_output_shapes
:@:@*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1
ќ
Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2Enter4PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/read*
_output_shapes
:@*#

frame_namewhile/while_context*
T0*
is_constant(*
parallel_iterations
А
Fwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_2SwitchEwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
:@:@

Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3Enter8PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:@*#

frame_namewhile/while_context
А
Fwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_3SwitchEwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
:@:@

4while/cond/PyramidFusedNet/fem_conv4/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

4while/cond/PyramidFusedNet/fem_conv4/LeakyRelu/alphaConst^while/cond/switch_t*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ё
2while/cond/PyramidFusedNet/fem_conv4/LeakyRelu/mulMul4while/cond/PyramidFusedNet/fem_conv4/LeakyRelu/alpha=while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
я
.while/cond/PyramidFusedNet/fem_conv4/LeakyReluMaximum2while/cond/PyramidFusedNet/fem_conv4/LeakyRelu/mul=while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
Ќ
=while/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"             *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
н
Awhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/depthwiseDepthwiseConv2dNativeJwhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/depthwise/Switch:1Jwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Switch_1:1*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`*
paddingSAME*
T0

Hwhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/depthwise/SwitchSwitchwhile/ResizeBilinearwhile/cond/pred_id*'
_class
loc:@while/ResizeBilinear*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ*
T0
А
7while/cond/PyramidFusedNet_1/fem_conv0/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/depthwise>while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Switch:1*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ *
paddingVALID*
T0
А
?while/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_1/fem_conv0/separable_conv2dFwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch:1Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_1:1Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_2:1Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_3:1*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ : : : : *
is_training( *
epsilon%o:*
T0

6while/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_1/fem_conv0/LeakyRelu/alphaConst^while/cond/switch_t*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_1/fem_conv0/LeakyRelu/mulMul6while/cond/PyramidFusedNet_1/fem_conv0/LeakyRelu/alpha?while/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ *
T0
ѕ
0while/cond/PyramidFusedNet_1/fem_conv0/LeakyReluMaximum4while/cond/PyramidFusedNet_1/fem_conv0/LeakyRelu/mul?while/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
Ќ
=while/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d/ShapeConst^while/cond/switch_t*
dtype0*
_output_shapes
:*%
valueB"             
Ќ
Ewhile/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d/dilation_rateConst^while/cond/switch_t*
_output_shapes
:*
valueB"      *
dtype0
С
Awhile/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_1/fem_conv0/LeakyReluHwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`
А
7while/cond/PyramidFusedNet_1/fem_conv1/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d/depthwise>while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
А
?while/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_1/fem_conv1/separable_conv2dFwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch:1Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_1:1Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_2:1Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_3:1*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( *
epsilon%o:*
T0

6while/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_1/fem_conv1/LeakyRelu/alphaConst^while/cond/switch_t*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_1/fem_conv1/LeakyRelu/mulMul6while/cond/PyramidFusedNet_1/fem_conv1/LeakyRelu/alpha?while/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
ѕ
0while/cond/PyramidFusedNet_1/fem_conv1/LeakyReluMaximum4while/cond/PyramidFusedNet_1/fem_conv1/LeakyRelu/mul?while/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
Ќ
=while/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"      @      *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
Т
Awhile/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_1/fem_conv1/LeakyReluHwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Switch:1*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР*
paddingSAME*
T0
А
7while/cond/PyramidFusedNet_1/fem_conv2/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d/depthwise>while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
А
?while/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_1/fem_conv2/separable_conv2dFwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch:1Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_1:1Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_2:1Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_3:1*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( 

6while/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_1/fem_conv2/LeakyRelu/alphaConst^while/cond/switch_t*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
ї
4while/cond/PyramidFusedNet_1/fem_conv2/LeakyRelu/mulMul6while/cond/PyramidFusedNet_1/fem_conv2/LeakyRelu/alpha?while/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
ѕ
0while/cond/PyramidFusedNet_1/fem_conv2/LeakyReluMaximum4while/cond/PyramidFusedNet_1/fem_conv2/LeakyRelu/mul?while/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
Ќ
=while/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"      @      *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
Т
Awhile/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_1/fem_conv2/LeakyReluHwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Switch:1*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР*
paddingSAME*
T0
А
7while/cond/PyramidFusedNet_1/fem_conv3/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d/depthwise>while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
А
?while/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_1/fem_conv3/separable_conv2dFwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch:1Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_1:1Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_2:1Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_3:1*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( 

6while/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_1/fem_conv3/LeakyRelu/alphaConst^while/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>
ї
4while/cond/PyramidFusedNet_1/fem_conv3/LeakyRelu/mulMul6while/cond/PyramidFusedNet_1/fem_conv3/LeakyRelu/alpha?while/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
ѕ
0while/cond/PyramidFusedNet_1/fem_conv3/LeakyReluMaximum4while/cond/PyramidFusedNet_1/fem_conv3/LeakyRelu/mul?while/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
Ќ
=while/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"      @      *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
Т
Awhile/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_1/fem_conv3/LeakyReluHwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР
А
7while/cond/PyramidFusedNet_1/fem_conv4/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d/depthwise>while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
А
?while/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_1/fem_conv4/separable_conv2dFwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch:1Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_1:1Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_2:1Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_3:1*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( 

6while/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_1/fem_conv4/LeakyRelu/alphaConst^while/cond/switch_t*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_1/fem_conv4/LeakyRelu/mulMul6while/cond/PyramidFusedNet_1/fem_conv4/LeakyRelu/alpha?while/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
ѕ
0while/cond/PyramidFusedNet_1/fem_conv4/LeakyReluMaximum4while/cond/PyramidFusedNet_1/fem_conv4/LeakyRelu/mul?while/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
w
while/cond/concat/axisConst^while/cond/switch_t*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
ф
while/cond/concatConcatV2.while/cond/PyramidFusedNet/fem_conv4/LeakyRelu0while/cond/PyramidFusedNet_1/fem_conv3/LeakyReluwhile/cond/concat/axis*
T0*
N*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџ
я
QPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights*%
valueB"            *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights*
valueB
 *xН*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights*
valueB
 *x=*
dtype0*
_output_shapes
: 
С
YPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/shape*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights*
dtype0*'
_output_shapes
:*
T0
о
OPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights*
_output_shapes
: 
љ
OPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights*'
_output_shapes
:
ы
KPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform/min*'
_output_shapes
:*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights
з
0PyramidFusedNet/dem0_log_conv0/depthwise_weights
VariableV2*
dtype0*'
_output_shapes
:*
shape:*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights
З
7PyramidFusedNet/dem0_log_conv0/depthwise_weights/AssignAssign0PyramidFusedNet/dem0_log_conv0/depthwise_weightsKPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights*'
_output_shapes
:
ъ
5PyramidFusedNet/dem0_log_conv0/depthwise_weights/readIdentity0PyramidFusedNet/dem0_log_conv0/depthwise_weights*'
_output_shapes
:*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights
я
QPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights*%
valueB"            *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights*
valueB
 *єєѕН*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights*
valueB
 *єєѕ=*
dtype0*
_output_shapes
: 
С
YPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights*
dtype0*'
_output_shapes
: 
о
OPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights
љ
OPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights*'
_output_shapes
: 
ы
KPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights*'
_output_shapes
: 
з
0PyramidFusedNet/dem0_log_conv0/pointwise_weights
VariableV2*
shape: *C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights*
dtype0*'
_output_shapes
: 
З
7PyramidFusedNet/dem0_log_conv0/pointwise_weights/AssignAssign0PyramidFusedNet/dem0_log_conv0/pointwise_weightsKPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights*'
_output_shapes
: 
ъ
5PyramidFusedNet/dem0_log_conv0/pointwise_weights/readIdentity0PyramidFusedNet/dem0_log_conv0/pointwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights*'
_output_shapes
: 
Б
Bwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"            *
dtype0*
_output_shapes
:
Б
Jwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
Џ
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwiseDepthwiseConv2dNativewhile/cond/concatOwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџ

Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/EnterEnter5PyramidFusedNet/dem0_log_conv0/depthwise_weights/read*'
_output_shapes
:*#

frame_namewhile/while_context*
T0*
is_constant(*
parallel_iterations
п
Mwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/Enter*:
_output_shapes(
&::
С
<while/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwiseEwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 

Bwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/EnterEnter5PyramidFusedNet/dem0_log_conv0/pointwise_weights/read*
T0*
is_constant(*
parallel_iterations*'
_output_shapes
: *#

frame_namewhile/while_context
С
Cwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Enterwhile/cond/pred_id*:
_output_shapes(
&: : *
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Enter
Я
?PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/Initializer/onesConst*A
_class7
53loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma*
valueB *  ?*
dtype0*
_output_shapes
: 
Й
.PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma
VariableV2*A
_class7
53loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma*
dtype0*
_output_shapes
: *
shape: 

5PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/AssignAssign.PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma?PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/Initializer/ones*
T0*A
_class7
53loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma*
_output_shapes
: 
з
3PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/readIdentity.PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma*
_output_shapes
: *
T0*A
_class7
53loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma
Ю
?PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/Initializer/zerosConst*@
_class6
42loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/beta*
valueB *    *
dtype0*
_output_shapes
: 
З
-PyramidFusedNet/dem0_log_conv0/BatchNorm/beta
VariableV2*@
_class6
42loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/beta*
dtype0*
_output_shapes
: *
shape: 

4PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/AssignAssign-PyramidFusedNet/dem0_log_conv0/BatchNorm/beta?PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/Initializer/zeros*
T0*@
_class6
42loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/beta*
_output_shapes
: 
д
2PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/readIdentity-PyramidFusedNet/dem0_log_conv0/BatchNorm/beta*
T0*@
_class6
42loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/beta*
_output_shapes
: 
м
FPyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean*
valueB *    *
dtype0*
_output_shapes
: 
Х
4PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean
VariableV2*G
_class=
;9loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *
shape: 
Б
;PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/AssignAssign4PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_meanFPyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/Initializer/zeros*
T0*G
_class=
;9loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean*
_output_shapes
: 
щ
9PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/readIdentity4PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean*
T0*G
_class=
;9loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean*
_output_shapes
: 
у
IPyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
: *K
_classA
?=loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance*
valueB *  ?
Э
8PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance
VariableV2*K
_classA
?=loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance*
dtype0*
_output_shapes
: *
shape: 
Р
?PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/AssignAssign8PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_varianceIPyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/Initializer/ones*
_output_shapes
: *
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance
ѕ
=PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/readIdentity8PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance*
_output_shapes
: *
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance
ж
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2dMwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch:1Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_1:1Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_2:1Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_3:1*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ : : : : *
is_training( *
epsilon%o:

Jwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/EnterEnter3PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
П
Kwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_1Enter2PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/read*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context*
T0*
is_constant(
Х
Mwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_2Enter9PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_3Enter=PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/read*
_output_shapes
: *#

frame_namewhile/while_context*
T0*
is_constant(*
parallel_iterations
Х
Mwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id* 
_output_shapes
: : *
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_3

;while/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/ConstConst^while/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *wО?

;while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyRelu/alphaConst^while/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>

9while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyRelu/mulMul;while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyRelu/alphaDwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 

5while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyReluMaximum9while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyRelu/mulDwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ *
T0
я
QPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights*%
valueB"             *
dtype0
й
OPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights*
valueB
 *OSО*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights*
valueB
 *OS>
Р
YPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights
о
OPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights
ј
OPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/sub*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights
ъ
KPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform/min*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights
е
0PyramidFusedNet/dem0_log_conv1/depthwise_weights
VariableV2*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights*
dtype0*&
_output_shapes
: *
shape: 
Ж
7PyramidFusedNet/dem0_log_conv1/depthwise_weights/AssignAssign0PyramidFusedNet/dem0_log_conv1/depthwise_weightsKPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights*&
_output_shapes
: 
щ
5PyramidFusedNet/dem0_log_conv1/depthwise_weights/readIdentity0PyramidFusedNet/dem0_log_conv1/depthwise_weights*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights
я
QPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights*%
valueB"      `      *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights*
valueB
 *б_}О*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights*
valueB
 *б_}>*
dtype0*
_output_shapes
: 
Р
YPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights*
dtype0*&
_output_shapes
:`
о
OPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights
ј
OPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/sub*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights*&
_output_shapes
:`*
T0
ъ
KPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform/min*&
_output_shapes
:`*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights
е
0PyramidFusedNet/dem0_log_conv1/pointwise_weights
VariableV2*
shape:`*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights*
dtype0*&
_output_shapes
:`
Ж
7PyramidFusedNet/dem0_log_conv1/pointwise_weights/AssignAssign0PyramidFusedNet/dem0_log_conv1/pointwise_weightsKPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights*&
_output_shapes
:`
щ
5PyramidFusedNet/dem0_log_conv1/pointwise_weights/readIdentity0PyramidFusedNet/dem0_log_conv1/pointwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights*&
_output_shapes
:`
Б
Bwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"             *
dtype0*
_output_shapes
:
Б
Jwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
в
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwiseDepthwiseConv2dNative5while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyReluOwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`

Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/EnterEnter5PyramidFusedNet/dem0_log_conv1/depthwise_weights/read*
parallel_iterations*&
_output_shapes
: *#

frame_namewhile/while_context*
T0*
is_constant(
н
Mwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/Enterwhile/cond/pred_id*8
_output_shapes&
$: : *
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/Enter
С
<while/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwiseEwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

Bwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/EnterEnter5PyramidFusedNet/dem0_log_conv1/pointwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
:`*#

frame_namewhile/while_context
П
Cwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Enterwhile/cond/pred_id*8
_output_shapes&
$:`:`*
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Enter
Я
?PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/Initializer/onesConst*A
_class7
53loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
Й
.PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes
:*
shape:*A
_class7
53loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma

5PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/AssignAssign.PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma?PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/Initializer/ones*
T0*A
_class7
53loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma*
_output_shapes
:
з
3PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/readIdentity.PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma*
_output_shapes
:*
T0*A
_class7
53loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma
Ю
?PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/Initializer/zerosConst*@
_class6
42loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
З
-PyramidFusedNet/dem0_log_conv1/BatchNorm/beta
VariableV2*@
_class6
42loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/beta*
dtype0*
_output_shapes
:*
shape:

4PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/AssignAssign-PyramidFusedNet/dem0_log_conv1/BatchNorm/beta?PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/Initializer/zeros*
T0*@
_class6
42loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/beta*
_output_shapes
:
д
2PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/readIdentity-PyramidFusedNet/dem0_log_conv1/BatchNorm/beta*
_output_shapes
:*
T0*@
_class6
42loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/beta
м
FPyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/Initializer/zerosConst*
_output_shapes
:*G
_class=
;9loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean*
valueB*    *
dtype0
Х
4PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean
VariableV2*G
_class=
;9loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean*
dtype0*
_output_shapes
:*
shape:
Б
;PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/AssignAssign4PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_meanFPyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/Initializer/zeros*
T0*G
_class=
;9loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean*
_output_shapes
:
щ
9PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/readIdentity4PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean*
_output_shapes
:*
T0*G
_class=
;9loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean
у
IPyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/Initializer/onesConst*K
_classA
?=loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes
:
Э
8PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance
VariableV2*K
_classA
?=loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance*
dtype0*
_output_shapes
:*
shape:
Р
?PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/AssignAssign8PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_varianceIPyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/Initializer/ones*
_output_shapes
:*
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance
ѕ
=PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/readIdentity8PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance*
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance*
_output_shapes
:
ж
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2dMwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch:1Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_1:1Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_2:1Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_3:1*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ::::*
is_training( *
epsilon%o:*
T0

Jwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/EnterEnter3PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context
П
Kwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
::

Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_1Enter2PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
::

Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_2Enter9PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
::

Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_3Enter=PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id* 
_output_shapes
::*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_3

;while/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

2while/cond/PyramidFusedNet_2/softmax/Reshape/shapeConst^while/cond/switch_t*
valueB"џџџџ   *
dtype0*
_output_shapes
:
у
,while/cond/PyramidFusedNet_2/softmax/ReshapeReshapeDwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm2while/cond/PyramidFusedNet_2/softmax/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0

,while/cond/PyramidFusedNet_2/softmax/SoftmaxSoftmax,while/cond/PyramidFusedNet_2/softmax/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0

*while/cond/PyramidFusedNet_2/softmax/ShapeShapeDwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm*
_output_shapes
:*
T0
ж
.while/cond/PyramidFusedNet_2/softmax/Reshape_1Reshape,while/cond/PyramidFusedNet_2/softmax/Softmax*while/cond/PyramidFusedNet_2/softmax/Shape*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

0while/cond/PyramidFusedNet_2/strided_slice/stackConst^while/cond/switch_t*%
valueB"               *
dtype0*
_output_shapes
:
Ё
2while/cond/PyramidFusedNet_2/strided_slice/stack_1Const^while/cond/switch_t*%
valueB"               *
dtype0*
_output_shapes
:
Ё
2while/cond/PyramidFusedNet_2/strided_slice/stack_2Const^while/cond/switch_t*%
valueB"            *
dtype0*
_output_shapes
:

*while/cond/PyramidFusedNet_2/strided_sliceStridedSlice.while/cond/PyramidFusedNet_2/softmax/Reshape_10while/cond/PyramidFusedNet_2/strided_slice/stack2while/cond/PyramidFusedNet_2/strided_slice/stack_12while/cond/PyramidFusedNet_2/strided_slice/stack_2*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
Index0*
T0*
shrink_axis_mask*

begin_mask*
end_mask
я
QPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights*%
valueB"             
й
OPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights*
valueB
 *OSО*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights*
valueB
 *OS>*
dtype0*
_output_shapes
: 
Р
YPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights*
dtype0*&
_output_shapes
: 
о
OPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights*
_output_shapes
: 
ј
OPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights*&
_output_shapes
: 
ъ
KPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform/min*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights
е
0PyramidFusedNet/dem0_reg_conv0/depthwise_weights
VariableV2*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights*
dtype0*&
_output_shapes
: *
shape: 
Ж
7PyramidFusedNet/dem0_reg_conv0/depthwise_weights/AssignAssign0PyramidFusedNet/dem0_reg_conv0/depthwise_weightsKPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights*&
_output_shapes
: 
щ
5PyramidFusedNet/dem0_reg_conv0/depthwise_weights/readIdentity0PyramidFusedNet/dem0_reg_conv0/depthwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights*&
_output_shapes
: 
я
QPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights*%
valueB"      `       *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights*
valueB
 *зГ]О*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights*
valueB
 *зГ]>*
dtype0*
_output_shapes
: 
Р
YPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights*
dtype0*&
_output_shapes
:` 
о
OPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights
ј
OPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/sub*&
_output_shapes
:` *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights
ъ
KPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights*&
_output_shapes
:` 
е
0PyramidFusedNet/dem0_reg_conv0/pointwise_weights
VariableV2*
dtype0*&
_output_shapes
:` *
shape:` *C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights
Ж
7PyramidFusedNet/dem0_reg_conv0/pointwise_weights/AssignAssign0PyramidFusedNet/dem0_reg_conv0/pointwise_weightsKPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights*&
_output_shapes
:` *
T0
щ
5PyramidFusedNet/dem0_reg_conv0/pointwise_weights/readIdentity0PyramidFusedNet/dem0_reg_conv0/pointwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights*&
_output_shapes
:` 
Б
Bwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/ShapeConst^while/cond/switch_t*
_output_shapes
:*%
valueB"             *
dtype0
Б
Jwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
в
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwiseDepthwiseConv2dNative5while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyReluOwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/Switch:1*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`*
paddingSAME*
T0*
strides


Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/EnterEnter5PyramidFusedNet/dem0_reg_conv0/depthwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
: *#

frame_namewhile/while_context
н
Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/Enter*8
_output_shapes&
$: : 
С
<while/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwiseEwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 

Bwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/EnterEnter5PyramidFusedNet/dem0_reg_conv0/pointwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
:` *#

frame_namewhile/while_context
П
Cwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Enterwhile/cond/pred_id*
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Enter*8
_output_shapes&
$:` :` 
Я
?PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/Initializer/onesConst*A
_class7
53loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma*
valueB *  ?*
dtype0*
_output_shapes
: 
Й
.PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma
VariableV2*A
_class7
53loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma*
dtype0*
_output_shapes
: *
shape: 

5PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/AssignAssign.PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma?PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/Initializer/ones*
T0*A
_class7
53loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma*
_output_shapes
: 
з
3PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/readIdentity.PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma*A
_class7
53loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma*
_output_shapes
: *
T0
Ю
?PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/Initializer/zerosConst*@
_class6
42loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta*
valueB *    *
dtype0*
_output_shapes
: 
З
-PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta
VariableV2*
shape: *@
_class6
42loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta*
dtype0*
_output_shapes
: 

4PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/AssignAssign-PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta?PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/Initializer/zeros*
T0*@
_class6
42loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta*
_output_shapes
: 
д
2PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/readIdentity-PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta*
T0*@
_class6
42loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta*
_output_shapes
: 
м
FPyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean*
valueB *    *
dtype0*
_output_shapes
: 
Х
4PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean
VariableV2*G
_class=
;9loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *
shape: 
Б
;PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/AssignAssign4PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_meanFPyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/Initializer/zeros*G
_class=
;9loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean*
_output_shapes
: *
T0
щ
9PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/readIdentity4PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean*
_output_shapes
: *
T0*G
_class=
;9loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean
у
IPyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/Initializer/onesConst*K
_classA
?=loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance*
valueB *  ?*
dtype0*
_output_shapes
: 
Э
8PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance
VariableV2*
shape: *K
_classA
?=loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 
Р
?PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/AssignAssign8PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_varianceIPyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/Initializer/ones*
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance*
_output_shapes
: 
ѕ
=PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/readIdentity8PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance*
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance*
_output_shapes
: 
ж
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2dMwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch:1Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1:1Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2:1Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3:1*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ : : : : *
is_training( *
epsilon%o:*
T0

Jwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/EnterEnter3PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
П
Kwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1Enter2PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2Enter9PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/read*
_output_shapes
: *#

frame_namewhile/while_context*
T0*
is_constant(*
parallel_iterations
Х
Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3Enter=PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
: : 

;while/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

;while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyRelu/alphaConst^while/cond/switch_t*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0

9while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyRelu/mulMul;while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyRelu/alphaDwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 

5while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyReluMaximum9while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyRelu/mulDwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
я
QPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights*%
valueB"             
й
OPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights*
valueB
 *OSО*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights*
valueB
 *OS>
Р
YPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights*
dtype0*&
_output_shapes
: 
о
OPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights*
_output_shapes
: *
T0
ј
OPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights*&
_output_shapes
: 
ъ
KPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform/min*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights
е
0PyramidFusedNet/dem0_reg_conv1/depthwise_weights
VariableV2*
shape: *C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights*
dtype0*&
_output_shapes
: 
Ж
7PyramidFusedNet/dem0_reg_conv1/depthwise_weights/AssignAssign0PyramidFusedNet/dem0_reg_conv1/depthwise_weightsKPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights*&
_output_shapes
: 
щ
5PyramidFusedNet/dem0_reg_conv1/depthwise_weights/readIdentity0PyramidFusedNet/dem0_reg_conv1/depthwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights*&
_output_shapes
: 
я
QPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights*%
valueB"      `      *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights*
valueB
 *чгzО*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights*
valueB
 *чгz>*
dtype0*
_output_shapes
: 
Р
YPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights*
dtype0*&
_output_shapes
:`
о
OPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights*
_output_shapes
: 
ј
OPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/sub*&
_output_shapes
:`*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights
ъ
KPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform/min*&
_output_shapes
:`*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights
е
0PyramidFusedNet/dem0_reg_conv1/pointwise_weights
VariableV2*
shape:`*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights*
dtype0*&
_output_shapes
:`
Ж
7PyramidFusedNet/dem0_reg_conv1/pointwise_weights/AssignAssign0PyramidFusedNet/dem0_reg_conv1/pointwise_weightsKPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights*&
_output_shapes
:`*
T0
щ
5PyramidFusedNet/dem0_reg_conv1/pointwise_weights/readIdentity0PyramidFusedNet/dem0_reg_conv1/pointwise_weights*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights*&
_output_shapes
:`*
T0
Б
Bwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"             *
dtype0*
_output_shapes
:
Б
Jwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
в
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwiseDepthwiseConv2dNative5while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyReluOwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/Switch:1*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`*
paddingSAME*
T0

Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/EnterEnter5PyramidFusedNet/dem0_reg_conv1/depthwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
: *#

frame_namewhile/while_context
н
Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/Enter*8
_output_shapes&
$: : 
С
<while/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwiseEwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Switch:1*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
paddingVALID

Bwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/EnterEnter5PyramidFusedNet/dem0_reg_conv1/pointwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
:`*#

frame_namewhile/while_context
П
Cwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Enterwhile/cond/pred_id*U
_classK
IGloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Enter*8
_output_shapes&
$:`:`*
T0
Я
?PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/Initializer/onesConst*A
_class7
53loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
Й
.PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma
VariableV2*
shape:*A
_class7
53loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma*
dtype0*
_output_shapes
:

5PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/AssignAssign.PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma?PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/Initializer/ones*
_output_shapes
:*
T0*A
_class7
53loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma
з
3PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/readIdentity.PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma*
_output_shapes
:*
T0*A
_class7
53loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma
Ю
?PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/Initializer/zerosConst*@
_class6
42loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
З
-PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta
VariableV2*@
_class6
42loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta*
dtype0*
_output_shapes
:*
shape:

4PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/AssignAssign-PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta?PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/Initializer/zeros*@
_class6
42loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta*
_output_shapes
:*
T0
д
2PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/readIdentity-PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta*
_output_shapes
:*
T0*@
_class6
42loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta
м
FPyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:*G
_class=
;9loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean*
valueB*    
Х
4PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes
:*
shape:*G
_class=
;9loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean
Б
;PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/AssignAssign4PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_meanFPyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/Initializer/zeros*
T0*G
_class=
;9loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean*
_output_shapes
:
щ
9PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/readIdentity4PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean*
T0*G
_class=
;9loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean*
_output_shapes
:
у
IPyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/Initializer/onesConst*K
_classA
?=loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes
:
Э
8PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance
VariableV2*
_output_shapes
:*
shape:*K
_classA
?=loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance*
dtype0
Р
?PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/AssignAssign8PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_varianceIPyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/Initializer/ones*
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance*
_output_shapes
:
ѕ
=PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/readIdentity8PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance*
_output_shapes
:*
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance
ж
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2dMwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch:1Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1:1Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2:1Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3:1*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ::::*
is_training( 

Jwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/EnterEnter3PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context
П
Kwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
::

Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1Enter2PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
::*
T0

Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2Enter9PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/read*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context*
T0*
is_constant(
Х
Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
::

Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3Enter=PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
::

;while/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 
я
QPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights*%
valueB"            *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/minConst*
_output_shapes
: *C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights*
valueB
 *xН*
dtype0
й
OPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights*
valueB
 *x=*
dtype0*
_output_shapes
: 
С
YPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/shape*'
_output_shapes
:*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights*
dtype0
о
OPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights*
_output_shapes
: 
љ
OPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights*'
_output_shapes
:
ы
KPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform/min*'
_output_shapes
:*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights
з
0PyramidFusedNet/dem1_log_conv0/depthwise_weights
VariableV2*
shape:*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights*
dtype0*'
_output_shapes
:
З
7PyramidFusedNet/dem1_log_conv0/depthwise_weights/AssignAssign0PyramidFusedNet/dem1_log_conv0/depthwise_weightsKPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform*'
_output_shapes
:*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights
ъ
5PyramidFusedNet/dem1_log_conv0/depthwise_weights/readIdentity0PyramidFusedNet/dem1_log_conv0/depthwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights*'
_output_shapes
:
я
QPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights*%
valueB"            
й
OPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights*
valueB
 *єєѕН*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights*
valueB
 *єєѕ=*
dtype0*
_output_shapes
: 
С
YPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights*
dtype0*'
_output_shapes
: 
о
OPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights
љ
OPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights*'
_output_shapes
: 
ы
KPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights*'
_output_shapes
: *
T0
з
0PyramidFusedNet/dem1_log_conv0/pointwise_weights
VariableV2*
shape: *C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights*
dtype0*'
_output_shapes
: 
З
7PyramidFusedNet/dem1_log_conv0/pointwise_weights/AssignAssign0PyramidFusedNet/dem1_log_conv0/pointwise_weightsKPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights*'
_output_shapes
: 
ъ
5PyramidFusedNet/dem1_log_conv0/pointwise_weights/readIdentity0PyramidFusedNet/dem1_log_conv0/pointwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights*'
_output_shapes
: 
Б
Bwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"            *
dtype0*
_output_shapes
:
Б
Jwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
Џ
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwiseDepthwiseConv2dNativewhile/cond/concatOwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџ

Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/EnterEnter5PyramidFusedNet/dem1_log_conv0/depthwise_weights/read*
T0*
is_constant(*
parallel_iterations*'
_output_shapes
:*#

frame_namewhile/while_context
п
Mwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enter*:
_output_shapes(
&::
С
<while/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwiseEwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Switch:1*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ *
paddingVALID*
T0

Bwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/EnterEnter5PyramidFusedNet/dem1_log_conv0/pointwise_weights/read*
T0*
is_constant(*
parallel_iterations*'
_output_shapes
: *#

frame_namewhile/while_context
С
Cwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enterwhile/cond/pred_id*
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enter*:
_output_shapes(
&: : 
Я
?PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/Initializer/onesConst*A
_class7
53loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma*
valueB *  ?*
dtype0*
_output_shapes
: 
Й
.PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma
VariableV2*A
_class7
53loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma*
dtype0*
_output_shapes
: *
shape: 

5PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/AssignAssign.PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma?PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/Initializer/ones*
T0*A
_class7
53loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma*
_output_shapes
: 
з
3PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/readIdentity.PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma*A
_class7
53loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma*
_output_shapes
: *
T0
Ю
?PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/Initializer/zerosConst*@
_class6
42loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/beta*
valueB *    *
dtype0*
_output_shapes
: 
З
-PyramidFusedNet/dem1_log_conv0/BatchNorm/beta
VariableV2*@
_class6
42loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/beta*
dtype0*
_output_shapes
: *
shape: 

4PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/AssignAssign-PyramidFusedNet/dem1_log_conv0/BatchNorm/beta?PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/Initializer/zeros*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/beta*
_output_shapes
: 
д
2PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/readIdentity-PyramidFusedNet/dem1_log_conv0/BatchNorm/beta*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/beta*
_output_shapes
: 
м
FPyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
: *G
_class=
;9loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean*
valueB *    
Х
4PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean
VariableV2*
shape: *G
_class=
;9loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean*
dtype0*
_output_shapes
: 
Б
;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/AssignAssign4PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_meanFPyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes
: *
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean
щ
9PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/readIdentity4PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean*
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean*
_output_shapes
: 
у
IPyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/Initializer/onesConst*K
_classA
?=loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance*
valueB *  ?*
dtype0*
_output_shapes
: 
Э
8PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance
VariableV2*
shape: *K
_classA
?=loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 
Р
?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/AssignAssign8PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_varianceIPyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/Initializer/ones*
_output_shapes
: *
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance
ѕ
=PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/readIdentity8PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance*
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance*
_output_shapes
: 
ж
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2dMwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch:1Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_1:1Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_2:1Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_3:1*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ : : : : *
is_training( 

Jwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/EnterEnter3PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
П
Kwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1Enter2PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2Enter9PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/read*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context*
T0*
is_constant(
Х
Mwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3Enter=PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
: : 

;while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

;while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyRelu/alphaConst^while/cond/switch_t*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 

9while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyRelu/mulMul;while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyRelu/alphaDwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 

5while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyReluMaximum9while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyRelu/mulDwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
я
QPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights*%
valueB"             *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights*
valueB
 *OSО*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/maxConst*
_output_shapes
: *C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights*
valueB
 *OS>*
dtype0
Р
YPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights*
dtype0*&
_output_shapes
: 
о
OPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights*
_output_shapes
: *
T0
ј
OPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/sub*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights
ъ
KPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights*&
_output_shapes
: *
T0
е
0PyramidFusedNet/dem1_log_conv1/depthwise_weights
VariableV2*
dtype0*&
_output_shapes
: *
shape: *C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights
Ж
7PyramidFusedNet/dem1_log_conv1/depthwise_weights/AssignAssign0PyramidFusedNet/dem1_log_conv1/depthwise_weightsKPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights
щ
5PyramidFusedNet/dem1_log_conv1/depthwise_weights/readIdentity0PyramidFusedNet/dem1_log_conv1/depthwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights*&
_output_shapes
: 
я
QPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*%
valueB"      `      *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*
valueB
 *б_}О*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*
valueB
 *б_}>*
dtype0*
_output_shapes
: 
Р
YPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*
dtype0*&
_output_shapes
:`
о
OPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*
_output_shapes
: 
ј
OPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*&
_output_shapes
:`
ъ
KPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*&
_output_shapes
:`
е
0PyramidFusedNet/dem1_log_conv1/pointwise_weights
VariableV2*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*
dtype0*&
_output_shapes
:`*
shape:`
Ж
7PyramidFusedNet/dem1_log_conv1/pointwise_weights/AssignAssign0PyramidFusedNet/dem1_log_conv1/pointwise_weightsKPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*&
_output_shapes
:`
щ
5PyramidFusedNet/dem1_log_conv1/pointwise_weights/readIdentity0PyramidFusedNet/dem1_log_conv1/pointwise_weights*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*&
_output_shapes
:`*
T0
Б
Bwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/ShapeConst^while/cond/switch_t*
dtype0*
_output_shapes
:*%
valueB"             
Б
Jwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/dilation_rateConst^while/cond/switch_t*
_output_shapes
:*
valueB"      *
dtype0
в
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwiseDepthwiseConv2dNative5while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyReluOwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`

Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/EnterEnter5PyramidFusedNet/dem1_log_conv1/depthwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
: *#

frame_namewhile/while_context
н
Mwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enter*8
_output_shapes&
$: : 
С
<while/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwiseEwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

Bwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/EnterEnter5PyramidFusedNet/dem1_log_conv1/pointwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
:`*#

frame_namewhile/while_context
П
Cwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enterwhile/cond/pred_id*
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enter*8
_output_shapes&
$:`:`
Я
?PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/Initializer/onesConst*A
_class7
53loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
Й
.PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes
:*
shape:*A
_class7
53loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma

5PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/AssignAssign.PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma?PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/Initializer/ones*
T0*A
_class7
53loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma*
_output_shapes
:
з
3PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/readIdentity.PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma*A
_class7
53loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma*
_output_shapes
:*
T0
Ю
?PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/beta*
valueB*    
З
-PyramidFusedNet/dem1_log_conv1/BatchNorm/beta
VariableV2*
shape:*@
_class6
42loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/beta*
dtype0*
_output_shapes
:

4PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/AssignAssign-PyramidFusedNet/dem1_log_conv1/BatchNorm/beta?PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/Initializer/zeros*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/beta*
_output_shapes
:
д
2PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/readIdentity-PyramidFusedNet/dem1_log_conv1/BatchNorm/beta*
_output_shapes
:*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/beta
м
FPyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
Х
4PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean
VariableV2*G
_class=
;9loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean*
dtype0*
_output_shapes
:*
shape:
Б
;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/AssignAssign4PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_meanFPyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/Initializer/zeros*
_output_shapes
:*
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean
щ
9PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/readIdentity4PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean*
_output_shapes
:*
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean
у
IPyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/Initializer/onesConst*K
_classA
?=loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes
:
Э
8PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance
VariableV2*K
_classA
?=loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance*
dtype0*
_output_shapes
:*
shape:
Р
?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/AssignAssign8PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_varianceIPyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/Initializer/ones*
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance*
_output_shapes
:
ѕ
=PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/readIdentity8PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance*
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance*
_output_shapes
:
ж
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2dMwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch:1Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_1:1Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_2:1Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_3:1*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ::::*
is_training( *
epsilon%o:*
T0

Jwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/EnterEnter3PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/read*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context*
T0
П
Kwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
::

Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1Enter2PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/read*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context*
T0
Х
Mwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id* 
_output_shapes
::*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1

Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2Enter9PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id* 
_output_shapes
::*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2

Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3Enter=PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/read*
_output_shapes
:*#

frame_namewhile/while_context*
T0*
is_constant(*
parallel_iterations
Х
Mwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
::

;while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

2while/cond/PyramidFusedNet_3/softmax/Reshape/shapeConst^while/cond/switch_t*
valueB"џџџџ   *
dtype0*
_output_shapes
:
у
,while/cond/PyramidFusedNet_3/softmax/ReshapeReshapeDwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm2while/cond/PyramidFusedNet_3/softmax/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0

,while/cond/PyramidFusedNet_3/softmax/SoftmaxSoftmax,while/cond/PyramidFusedNet_3/softmax/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0

*while/cond/PyramidFusedNet_3/softmax/ShapeShapeDwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm*
_output_shapes
:*
T0
ж
.while/cond/PyramidFusedNet_3/softmax/Reshape_1Reshape,while/cond/PyramidFusedNet_3/softmax/Softmax*while/cond/PyramidFusedNet_3/softmax/Shape*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

0while/cond/PyramidFusedNet_3/strided_slice/stackConst^while/cond/switch_t*%
valueB"               *
dtype0*
_output_shapes
:
Ё
2while/cond/PyramidFusedNet_3/strided_slice/stack_1Const^while/cond/switch_t*%
valueB"               *
dtype0*
_output_shapes
:
Ё
2while/cond/PyramidFusedNet_3/strided_slice/stack_2Const^while/cond/switch_t*
dtype0*
_output_shapes
:*%
valueB"            

*while/cond/PyramidFusedNet_3/strided_sliceStridedSlice.while/cond/PyramidFusedNet_3/softmax/Reshape_10while/cond/PyramidFusedNet_3/strided_slice/stack2while/cond/PyramidFusedNet_3/strided_slice/stack_12while/cond/PyramidFusedNet_3/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
T0*
Index0
я
QPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights*%
valueB"             *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights*
valueB
 *OSО*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights*
valueB
 *OS>*
dtype0*
_output_shapes
: 
Р
YPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights*
dtype0*&
_output_shapes
: 
о
OPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights*
_output_shapes
: 
ј
OPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/sub*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights
ъ
KPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights*&
_output_shapes
: 
е
0PyramidFusedNet/dem1_reg_conv0/depthwise_weights
VariableV2*
dtype0*&
_output_shapes
: *
shape: *C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights
Ж
7PyramidFusedNet/dem1_reg_conv0/depthwise_weights/AssignAssign0PyramidFusedNet/dem1_reg_conv0/depthwise_weightsKPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights*&
_output_shapes
: 
щ
5PyramidFusedNet/dem1_reg_conv0/depthwise_weights/readIdentity0PyramidFusedNet/dem1_reg_conv0/depthwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights*&
_output_shapes
: 
я
QPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*%
valueB"      `       *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*
valueB
 *зГ]О*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*
valueB
 *зГ]>*
dtype0*
_output_shapes
: 
Р
YPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*
dtype0*&
_output_shapes
:` 
о
OPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*
_output_shapes
: 
ј
OPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*&
_output_shapes
:` 
ъ
KPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*&
_output_shapes
:` 
е
0PyramidFusedNet/dem1_reg_conv0/pointwise_weights
VariableV2*
shape:` *C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*
dtype0*&
_output_shapes
:` 
Ж
7PyramidFusedNet/dem1_reg_conv0/pointwise_weights/AssignAssign0PyramidFusedNet/dem1_reg_conv0/pointwise_weightsKPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*&
_output_shapes
:` 
щ
5PyramidFusedNet/dem1_reg_conv0/pointwise_weights/readIdentity0PyramidFusedNet/dem1_reg_conv0/pointwise_weights*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*&
_output_shapes
:` *
T0
Б
Bwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"             *
dtype0*
_output_shapes
:
Б
Jwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
в
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwiseDepthwiseConv2dNative5while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyReluOwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`

Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/EnterEnter5PyramidFusedNet/dem1_reg_conv0/depthwise_weights/read*&
_output_shapes
: *#

frame_namewhile/while_context*
T0*
is_constant(*
parallel_iterations
н
Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enterwhile/cond/pred_id*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enter*8
_output_shapes&
$: : *
T0
С
<while/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwiseEwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 

Bwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/EnterEnter5PyramidFusedNet/dem1_reg_conv0/pointwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
:` *#

frame_namewhile/while_context
П
Cwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enterwhile/cond/pred_id*
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enter*8
_output_shapes&
$:` :` 
Я
?PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma*
valueB *  ?
Й
.PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes
: *
shape: *A
_class7
53loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma

5PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/AssignAssign.PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma?PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/Initializer/ones*
T0*A
_class7
53loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma*
_output_shapes
: 
з
3PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/readIdentity.PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma*
T0*A
_class7
53loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma*
_output_shapes
: 
Ю
?PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta*
valueB *    
З
-PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta
VariableV2*
_output_shapes
: *
shape: *@
_class6
42loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta*
dtype0

4PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/AssignAssign-PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta?PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/Initializer/zeros*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta*
_output_shapes
: 
д
2PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/readIdentity-PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta*
_output_shapes
: 
м
FPyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean*
valueB *    *
dtype0*
_output_shapes
: 
Х
4PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean
VariableV2*G
_class=
;9loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean*
dtype0*
_output_shapes
: *
shape: 
Б
;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/AssignAssign4PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_meanFPyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/Initializer/zeros*
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean*
_output_shapes
: 
щ
9PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/readIdentity4PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean*
_output_shapes
: *
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean
у
IPyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/Initializer/onesConst*K
_classA
?=loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance*
valueB *  ?*
dtype0*
_output_shapes
: 
Э
8PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance
VariableV2*
shape: *K
_classA
?=loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance*
dtype0*
_output_shapes
: 
Р
?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/AssignAssign8PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_varianceIPyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/Initializer/ones*
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance*
_output_shapes
: 
ѕ
=PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/readIdentity8PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance*
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance*
_output_shapes
: 
ж
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2dMwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch:1Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1:1Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2:1Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3:1*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ : : : : *
is_training( *
epsilon%o:*
T0

Jwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/EnterEnter3PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
П
Kwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1Enter2PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2Enter9PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
: : 

Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3Enter=PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/read*
parallel_iterations*
_output_shapes
: *#

frame_namewhile/while_context*
T0*
is_constant(
Х
Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
: : 

;while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 

;while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyRelu/alphaConst^while/cond/switch_t*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 

9while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyRelu/mulMul;while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyRelu/alphaDwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 

5while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyReluMaximum9while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyRelu/mulDwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
я
QPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights*%
valueB"             *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights*
valueB
 *OSО*
dtype0*
_output_shapes
: 
й
OPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights*
valueB
 *OS>*
dtype0*
_output_shapes
: 
Р
YPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/shape*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights*
dtype0*&
_output_shapes
: *
T0
о
OPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights*
_output_shapes
: 
ј
OPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/sub*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights
ъ
KPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights*&
_output_shapes
: 
е
0PyramidFusedNet/dem1_reg_conv1/depthwise_weights
VariableV2*
shape: *C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights*
dtype0*&
_output_shapes
: 
Ж
7PyramidFusedNet/dem1_reg_conv1/depthwise_weights/AssignAssign0PyramidFusedNet/dem1_reg_conv1/depthwise_weightsKPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights*&
_output_shapes
: 
щ
5PyramidFusedNet/dem1_reg_conv1/depthwise_weights/readIdentity0PyramidFusedNet/dem1_reg_conv1/depthwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights*&
_output_shapes
: 
я
QPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights*%
valueB"      `      *
dtype0*
_output_shapes
:
й
OPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/minConst*
_output_shapes
: *C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights*
valueB
 *чгzО*
dtype0
й
OPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights*
valueB
 *чгz>*
dtype0*
_output_shapes
: 
Р
YPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:`*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights
о
OPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/subSubOPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/maxOPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights*
_output_shapes
: 
ј
OPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/mulMulYPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/RandomUniformOPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/sub*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights*&
_output_shapes
:`*
T0
ъ
KPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniformAddOPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/mulOPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights*&
_output_shapes
:`
е
0PyramidFusedNet/dem1_reg_conv1/pointwise_weights
VariableV2*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights*
dtype0*&
_output_shapes
:`*
shape:`
Ж
7PyramidFusedNet/dem1_reg_conv1/pointwise_weights/AssignAssign0PyramidFusedNet/dem1_reg_conv1/pointwise_weightsKPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights*&
_output_shapes
:`
щ
5PyramidFusedNet/dem1_reg_conv1/pointwise_weights/readIdentity0PyramidFusedNet/dem1_reg_conv1/pointwise_weights*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights*&
_output_shapes
:`
Б
Bwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/ShapeConst^while/cond/switch_t*%
valueB"             *
dtype0*
_output_shapes
:
Б
Jwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/dilation_rateConst^while/cond/switch_t*
valueB"      *
dtype0*
_output_shapes
:
в
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwiseDepthwiseConv2dNative5while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyReluOwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Switch:1*
paddingSAME*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`

Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/EnterEnter5PyramidFusedNet/dem1_reg_conv1/depthwise_weights/read*
is_constant(*
parallel_iterations*&
_output_shapes
: *#

frame_namewhile/while_context*
T0
н
Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enter*8
_output_shapes&
$: : 
С
<while/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwiseEwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Switch:1*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

Bwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/EnterEnter5PyramidFusedNet/dem1_reg_conv1/pointwise_weights/read*
T0*
is_constant(*
parallel_iterations*&
_output_shapes
:`*#

frame_namewhile/while_context
П
Cwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enterwhile/cond/pred_id*
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enter*8
_output_shapes&
$:`:`
Я
?PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/Initializer/onesConst*A
_class7
53loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
Й
.PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma
VariableV2*
_output_shapes
:*
shape:*A
_class7
53loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma*
dtype0

5PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/AssignAssign.PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma?PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/Initializer/ones*
T0*A
_class7
53loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma*
_output_shapes
:
з
3PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/readIdentity.PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma*
T0*A
_class7
53loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma*
_output_shapes
:
Ю
?PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta*
valueB*    
З
-PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta
VariableV2*
shape:*@
_class6
42loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta*
dtype0*
_output_shapes
:

4PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/AssignAssign-PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta?PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/Initializer/zeros*
_output_shapes
:*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta
д
2PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/readIdentity-PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta*
_output_shapes
:
м
FPyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
Х
4PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes
:*
shape:*G
_class=
;9loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean
Б
;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/AssignAssign4PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_meanFPyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/Initializer/zeros*
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean*
_output_shapes
:
щ
9PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/readIdentity4PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean*
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean*
_output_shapes
:
у
IPyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/Initializer/onesConst*
_output_shapes
:*K
_classA
?=loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance*
valueB*  ?*
dtype0
Э
8PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance
VariableV2*K
_classA
?=loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance*
dtype0*
_output_shapes
:*
shape:
Р
?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/AssignAssign8PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_varianceIPyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/Initializer/ones*
_output_shapes
:*
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance
ѕ
=PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/readIdentity8PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance*
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance*
_output_shapes
:
ж
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2dMwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch:1Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1:1Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2:1Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3:1*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ::::*
is_training( 

Jwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/EnterEnter3PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context
П
Kwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id* 
_output_shapes
::*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter

Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1Enter2PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/read*
_output_shapes
:*#

frame_namewhile/while_context*
T0*
is_constant(*
parallel_iterations
Х
Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id* 
_output_shapes
::*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1

Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2Enter9PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/read*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context*
T0
Х
Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id* 
_output_shapes
::*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2

Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3Enter=PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/read*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*#

frame_namewhile/while_context
Х
Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
::

;while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/ConstConst^while/cond/switch_t*
valueB
 *wО?*
dtype0*
_output_shapes
: 
t
while/cond/batch_decode/RankConst^while/cond/switch_t*
dtype0*
_output_shapes
: *
value	B :
~
&while/cond/batch_decode/assert_equal/yConst^while/cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 

*while/cond/batch_decode/assert_equal/EqualEqualwhile/cond/batch_decode/Rank&while/cond/batch_decode/assert_equal/y*
T0*
_output_shapes
: 

*while/cond/batch_decode/assert_equal/ConstConst^while/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 

(while/cond/batch_decode/assert_equal/AllAll*while/cond/batch_decode/assert_equal/Equal*while/cond/batch_decode/assert_equal/Const*
_output_shapes
: 

1while/cond/batch_decode/assert_equal/Assert/ConstConst^while/cond/switch_t*
valueB B *
dtype0*
_output_shapes
: 
Е
3while/cond/batch_decode/assert_equal/Assert/Const_1Const^while/cond/switch_t*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
Џ
3while/cond/batch_decode/assert_equal/Assert/Const_2Const^while/cond/switch_t*
_output_shapes
: *6
value-B+ B%x (while/cond/batch_decode/Rank:0) = *
dtype0
Й
3while/cond/batch_decode/assert_equal/Assert/Const_3Const^while/cond/switch_t*
_output_shapes
: *@
value7B5 B/y (while/cond/batch_decode/assert_equal/y:0) = *
dtype0

9while/cond/batch_decode/assert_equal/Assert/Assert/data_0Const^while/cond/switch_t*
valueB B *
dtype0*
_output_shapes
: 
Л
9while/cond/batch_decode/assert_equal/Assert/Assert/data_1Const^while/cond/switch_t*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
Е
9while/cond/batch_decode/assert_equal/Assert/Assert/data_2Const^while/cond/switch_t*6
value-B+ B%x (while/cond/batch_decode/Rank:0) = *
dtype0*
_output_shapes
: 
П
9while/cond/batch_decode/assert_equal/Assert/Assert/data_4Const^while/cond/switch_t*@
value7B5 B/y (while/cond/batch_decode/assert_equal/y:0) = *
dtype0*
_output_shapes
: 
Љ
2while/cond/batch_decode/assert_equal/Assert/AssertAssert(while/cond/batch_decode/assert_equal/All9while/cond/batch_decode/assert_equal/Assert/Assert/data_09while/cond/batch_decode/assert_equal/Assert/Assert/data_19while/cond/batch_decode/assert_equal/Assert/Assert/data_2while/cond/batch_decode/Rank9while/cond/batch_decode/assert_equal/Assert/Assert/data_4&while/cond/batch_decode/assert_equal/y*
T

2
v
while/cond/batch_decode/Rank_1Const^while/cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 

(while/cond/batch_decode/assert_equal_1/yConst^while/cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
 
,while/cond/batch_decode/assert_equal_1/EqualEqualwhile/cond/batch_decode/Rank_1(while/cond/batch_decode/assert_equal_1/y*
_output_shapes
: *
T0

,while/cond/batch_decode/assert_equal_1/ConstConst^while/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
Ѕ
*while/cond/batch_decode/assert_equal_1/AllAll,while/cond/batch_decode/assert_equal_1/Equal,while/cond/batch_decode/assert_equal_1/Const*
_output_shapes
: 

3while/cond/batch_decode/assert_equal_1/Assert/ConstConst^while/cond/switch_t*
valueB B *
dtype0*
_output_shapes
: 
З
5while/cond/batch_decode/assert_equal_1/Assert/Const_1Const^while/cond/switch_t*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
Г
5while/cond/batch_decode/assert_equal_1/Assert/Const_2Const^while/cond/switch_t*8
value/B- B'x (while/cond/batch_decode/Rank_1:0) = *
dtype0*
_output_shapes
: 
Н
5while/cond/batch_decode/assert_equal_1/Assert/Const_3Const^while/cond/switch_t*B
value9B7 B1y (while/cond/batch_decode/assert_equal_1/y:0) = *
dtype0*
_output_shapes
: 

;while/cond/batch_decode/assert_equal_1/Assert/Assert/data_0Const^while/cond/switch_t*
valueB B *
dtype0*
_output_shapes
: 
Н
;while/cond/batch_decode/assert_equal_1/Assert/Assert/data_1Const^while/cond/switch_t*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
Й
;while/cond/batch_decode/assert_equal_1/Assert/Assert/data_2Const^while/cond/switch_t*8
value/B- B'x (while/cond/batch_decode/Rank_1:0) = *
dtype0*
_output_shapes
: 
У
;while/cond/batch_decode/assert_equal_1/Assert/Assert/data_4Const^while/cond/switch_t*B
value9B7 B1y (while/cond/batch_decode/assert_equal_1/y:0) = *
dtype0*
_output_shapes
: 
Й
4while/cond/batch_decode/assert_equal_1/Assert/AssertAssert*while/cond/batch_decode/assert_equal_1/All;while/cond/batch_decode/assert_equal_1/Assert/Assert/data_0;while/cond/batch_decode/assert_equal_1/Assert/Assert/data_1;while/cond/batch_decode/assert_equal_1/Assert/Assert/data_2while/cond/batch_decode/Rank_1;while/cond/batch_decode/assert_equal_1/Assert/Assert/data_4(while/cond/batch_decode/assert_equal_1/y*
T

2
ц
%while/cond/batch_decode/Reshape/shapeConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
_output_shapes
:*!
valueB"   џџџџ   *
dtype0
Э
while/cond/batch_decode/ReshapeReshapeDwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm%while/cond/batch_decode/Reshape/shape*+
_output_shapes
:џџџџџџџџџ*
T0
ф
'while/cond/batch_decode/Reshape_1/shapeConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB"   џџџџ
Г
!while/cond/batch_decode/Reshape_1Reshape*while/cond/PyramidFusedNet_2/strided_slice'while/cond/batch_decode/Reshape_1/shape*
T0*'
_output_shapes
:џџџџџџџџџ

while/cond/batch_decode/unstackUnpackwhile/cond/batch_decode/Reshape*	
num*
T0*'
_output_shapes
:џџџџџџџџџ
М
Nwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/RankRankWwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch:13^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
T0*
_output_shapes
: 
щ
Uwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Rank/SwitchSwitchwhile/Reshapewhile/cond/pred_id* 
_class
loc:@while/Reshape*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
T0
§
Owhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/sub/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 

Mwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/subSubNwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/RankOwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/sub/y*
T0*
_output_shapes
: 

Uwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range/startConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 

Uwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range/deltaConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
ћ
Owhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/RangeRangeUwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range/startNwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/RankUwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range/delta*#
_output_shapes
:џџџџџџџџџ
Є
Owhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/sub_1SubMwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/subOwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range*#
_output_shapes
:џџџџџџџџџ*
T0
В
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose	TransposeWwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch:1Owhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/sub_1*
T0*'
_output_shapes
:џџџџџџџџџ

Gwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstackUnpackIwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose*	
num*
T0*P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ

Cwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/subSubIwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:3Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:1*#
_output_shapes
:џџџџџџџџџ*
T0

Ewhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/sub_1SubIwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:2Gwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack*#
_output_shapes
:џџџџџџџџџ*
T0
њ
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 

Gwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truedivRealDivEwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/sub_1Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv/y*#
_output_shapes
:џџџџџџџџџ*
T0

Cwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/addAddGwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstackGwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv*
T0*#
_output_shapes
:џџџџџџџџџ
ќ
Kwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv_1/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 

Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv_1RealDivCwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/subKwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv_1/y*
T0*#
_output_shapes
:џџџџџџџџџ

Ewhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/add_1AddIwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:1Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ
w
-while/cond/batch_decode/Decode/transpose/RankRankwhile/cond/batch_decode/unstack*
T0*
_output_shapes
: 
м
.while/cond/batch_decode/Decode/transpose/sub/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
Г
,while/cond/batch_decode/Decode/transpose/subSub-while/cond/batch_decode/Decode/transpose/Rank.while/cond/batch_decode/Decode/transpose/sub/y*
_output_shapes
: *
T0
т
4while/cond/batch_decode/Decode/transpose/Range/startConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
value	B : 
т
4while/cond/batch_decode/Decode/transpose/Range/deltaConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
ї
.while/cond/batch_decode/Decode/transpose/RangeRange4while/cond/batch_decode/Decode/transpose/Range/start-while/cond/batch_decode/Decode/transpose/Rank4while/cond/batch_decode/Decode/transpose/Range/delta*#
_output_shapes
:џџџџџџџџџ
С
.while/cond/batch_decode/Decode/transpose/sub_1Sub,while/cond/batch_decode/Decode/transpose/sub.while/cond/batch_decode/Decode/transpose/Range*
T0*#
_output_shapes
:џџџџџџџџџ
И
(while/cond/batch_decode/Decode/transpose	Transposewhile/cond/batch_decode/unstack.while/cond/batch_decode/Decode/transpose/sub_1*'
_output_shapes
:џџџџџџџџџ*
T0
Р
&while/cond/batch_decode/Decode/unstackUnpack(while/cond/batch_decode/Decode/transpose*	
num*
T0*P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ

"while/cond/batch_decode/Decode/ExpExp(while/cond/batch_decode/Decode/unstack:3*#
_output_shapes
:џџџџџџџџџ*
T0
Р
"while/cond/batch_decode/Decode/mulMul"while/cond/batch_decode/Decode/ExpCwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/sub*
T0*#
_output_shapes
:џџџџџџџџџ

$while/cond/batch_decode/Decode/Exp_1Exp(while/cond/batch_decode/Decode/unstack:2*#
_output_shapes
:џџџџџџџџџ*
T0
Ц
$while/cond/batch_decode/Decode/mul_1Mul$while/cond/batch_decode/Decode/Exp_1Ewhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/sub_1*
T0*#
_output_shapes
:џџџџџџџџџ
Ш
$while/cond/batch_decode/Decode/mul_2Mul&while/cond/batch_decode/Decode/unstackEwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/sub_1*
T0*#
_output_shapes
:џџџџџџџџџ
Т
"while/cond/batch_decode/Decode/addAdd$while/cond/batch_decode/Decode/mul_2Cwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/add*#
_output_shapes
:џџџџџџџџџ*
T0
Ш
$while/cond/batch_decode/Decode/mul_3Mul(while/cond/batch_decode/Decode/unstack:1Cwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/sub*#
_output_shapes
:џџџџџџџџџ*
T0
Ц
$while/cond/batch_decode/Decode/add_1Add$while/cond/batch_decode/Decode/mul_3Ewhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/add_1*#
_output_shapes
:џџџџџџџџџ*
T0
й
(while/cond/batch_decode/Decode/truediv/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 
Џ
&while/cond/batch_decode/Decode/truedivRealDiv$while/cond/batch_decode/Decode/mul_1(while/cond/batch_decode/Decode/truediv/y*
T0*#
_output_shapes
:џџџџџџџџџ
Ѓ
"while/cond/batch_decode/Decode/subSub"while/cond/batch_decode/Decode/add&while/cond/batch_decode/Decode/truediv*#
_output_shapes
:џџџџџџџџџ*
T0
л
*while/cond/batch_decode/Decode/truediv_1/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
_output_shapes
: *
valueB
 *   @*
dtype0
Б
(while/cond/batch_decode/Decode/truediv_1RealDiv"while/cond/batch_decode/Decode/mul*while/cond/batch_decode/Decode/truediv_1/y*
T0*#
_output_shapes
:џџџџџџџџџ
Љ
$while/cond/batch_decode/Decode/sub_1Sub$while/cond/batch_decode/Decode/add_1(while/cond/batch_decode/Decode/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ
л
*while/cond/batch_decode/Decode/truediv_2/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 
Г
(while/cond/batch_decode/Decode/truediv_2RealDiv$while/cond/batch_decode/Decode/mul_1*while/cond/batch_decode/Decode/truediv_2/y*#
_output_shapes
:џџџџџџџџџ*
T0
Ї
$while/cond/batch_decode/Decode/add_2Add"while/cond/batch_decode/Decode/add(while/cond/batch_decode/Decode/truediv_2*#
_output_shapes
:џџџџџџџџџ*
T0
л
*while/cond/batch_decode/Decode/truediv_3/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
valueB
 *   @
Б
(while/cond/batch_decode/Decode/truediv_3RealDiv"while/cond/batch_decode/Decode/mul*while/cond/batch_decode/Decode/truediv_3/y*#
_output_shapes
:џџџџџџџџџ*
T0
Љ
$while/cond/batch_decode/Decode/add_3Add$while/cond/batch_decode/Decode/add_1(while/cond/batch_decode/Decode/truediv_3*#
_output_shapes
:џџџџџџџџџ*
T0
§
$while/cond/batch_decode/Decode/stackPack"while/cond/batch_decode/Decode/sub$while/cond/batch_decode/Decode/sub_1$while/cond/batch_decode/Decode/add_2$while/cond/batch_decode/Decode/add_3*
N*'
_output_shapes
:џџџџџџџџџ*
T0
~
/while/cond/batch_decode/Decode/transpose_1/RankRank$while/cond/batch_decode/Decode/stack*
T0*
_output_shapes
: 
о
0while/cond/batch_decode/Decode/transpose_1/sub/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
Й
.while/cond/batch_decode/Decode/transpose_1/subSub/while/cond/batch_decode/Decode/transpose_1/Rank0while/cond/batch_decode/Decode/transpose_1/sub/y*
T0*
_output_shapes
: 
ф
6while/cond/batch_decode/Decode/transpose_1/Range/startConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
ф
6while/cond/batch_decode/Decode/transpose_1/Range/deltaConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
џ
0while/cond/batch_decode/Decode/transpose_1/RangeRange6while/cond/batch_decode/Decode/transpose_1/Range/start/while/cond/batch_decode/Decode/transpose_1/Rank6while/cond/batch_decode/Decode/transpose_1/Range/delta*#
_output_shapes
:џџџџџџџџџ
Ч
0while/cond/batch_decode/Decode/transpose_1/sub_1Sub.while/cond/batch_decode/Decode/transpose_1/sub0while/cond/batch_decode/Decode/transpose_1/Range*#
_output_shapes
:џџџџџџџџџ*
T0
С
*while/cond/batch_decode/Decode/transpose_1	Transpose$while/cond/batch_decode/Decode/stack0while/cond/batch_decode/Decode/transpose_1/sub_1*
T0*'
_output_shapes
:џџџџџџџџџ

while/cond/batch_decode/stackPack*while/cond/batch_decode/Decode/transpose_1*
T0*
N*+
_output_shapes
:џџџџџџџџџ
ы
5while/cond/batch_decode/nms_batch/strided_slice/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
э
7while/cond/batch_decode/nms_batch/strided_slice/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
э
7while/cond/batch_decode/nms_batch/strided_slice/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
о
/while/cond/batch_decode/nms_batch/strided_sliceStridedSlicewhile/cond/batch_decode/stack5while/cond/batch_decode/nms_batch/strided_slice/stack7while/cond/batch_decode/nms_batch/strided_slice/stack_17while/cond/batch_decode/nms_batch/strided_slice/stack_2*
T0*
Index0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
э
7while/cond/batch_decode/nms_batch/strided_slice_1/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB: 
я
9while/cond/batch_decode/nms_batch/strided_slice_1/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
я
9while/cond/batch_decode/nms_batch/strided_slice_1/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
ц
1while/cond/batch_decode/nms_batch/strided_slice_1StridedSlice!while/cond/batch_decode/Reshape_17while/cond/batch_decode/nms_batch/strided_slice_1/stack9while/cond/batch_decode/nms_batch/strided_slice_1/stack_19while/cond/batch_decode/nms_batch/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ
ф
3while/cond/batch_decode/nms_batch/nms/iou_thresholdConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 
ц
5while/cond/batch_decode/nms_batch/nms/score_thresholdConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
а
-while/cond/batch_decode/nms_batch/nms/GreaterGreater1while/cond/batch_decode/nms_batch/strided_slice_15while/cond/batch_decode/nms_batch/nms/score_threshold*
T0*#
_output_shapes
:џџџџџџџџџ

8while/cond/batch_decode/nms_batch/nms/boolean_mask/ShapeShape/while/cond/batch_decode/nms_batch/strided_slice*
_output_shapes
:*
T0
ќ
Fwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ў
Hwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
ў
Hwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

@while/cond/batch_decode/nms_batch/nms/boolean_mask/strided_sliceStridedSlice8while/cond/batch_decode/nms_batch/nms/boolean_mask/ShapeFwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stackHwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stack_1Hwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stack_2*
_output_shapes
:*
Index0*
T0
џ
Iwhile/cond/batch_decode/nms_batch/nms/boolean_mask/Prod/reduction_indicesConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
э
7while/cond/batch_decode/nms_batch/nms/boolean_mask/ProdProd@while/cond/batch_decode/nms_batch/nms/boolean_mask/strided_sliceIwhile/cond/batch_decode/nms_batch/nms/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0

:while/cond/batch_decode/nms_batch/nms/boolean_mask/Shape_1Shape/while/cond/batch_decode/nms_batch/strided_slice*
T0*
_output_shapes
:
ў
Hwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB: 

Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
В
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1StridedSlice:while/cond/batch_decode/nms_batch/nms/boolean_mask/Shape_1Hwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stackJwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stack_1Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stack_2*
T0*
Index0*

begin_mask*
_output_shapes
: 

:while/cond/batch_decode/nms_batch/nms/boolean_mask/Shape_2Shape/while/cond/batch_decode/nms_batch/strided_slice*
T0*
_output_shapes
:
ў
Hwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
В
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2StridedSlice:while/cond/batch_decode/nms_batch/nms/boolean_mask/Shape_2Hwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stackJwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stack_1Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stack_2*
end_mask*
_output_shapes
:*
T0*
Index0
Б
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/concat/values_1Pack7while/cond/batch_decode/nms_batch/nms/boolean_mask/Prod*
T0*
N*
_output_shapes
:
ь
>while/cond/batch_decode/nms_batch/nms/boolean_mask/concat/axisConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
џ
9while/cond/batch_decode/nms_batch/nms/boolean_mask/concatConcatV2Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/concat/values_1Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2>while/cond/batch_decode/nms_batch/nms/boolean_mask/concat/axis*
T0*
N*
_output_shapes
:
у
:while/cond/batch_decode/nms_batch/nms/boolean_mask/ReshapeReshape/while/cond/batch_decode/nms_batch/strided_slice9while/cond/batch_decode/nms_batch/nms/boolean_mask/concat*
T0*'
_output_shapes
:џџџџџџџџџ

Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape_1/shapeConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
ш
<while/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape_1Reshape-while/cond/batch_decode/nms_batch/nms/GreaterBwhile/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape_1/shape*
T0
*#
_output_shapes
:џџџџџџџџџ
Ј
8while/cond/batch_decode/nms_batch/nms/boolean_mask/WhereWhere<while/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ф
:while/cond/batch_decode/nms_batch/nms/boolean_mask/SqueezeSqueeze8while/cond/batch_decode/nms_batch/nms/boolean_mask/Where*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims

ю
@while/cond/batch_decode/nms_batch/nms/boolean_mask/GatherV2/axisConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
_output_shapes
: *
value	B : *
dtype0
ж
;while/cond/batch_decode/nms_batch/nms/boolean_mask/GatherV2GatherV2:while/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape:while/cond/batch_decode/nms_batch/nms/boolean_mask/Squeeze@while/cond/batch_decode/nms_batch/nms/boolean_mask/GatherV2/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0

:while/cond/batch_decode/nms_batch/nms/boolean_mask_1/ShapeShape1while/cond/batch_decode/nms_batch/strided_slice_1*
_output_shapes
:*
T0
ў
Hwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB: 

Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
Ђ
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_sliceStridedSlice:while/cond/batch_decode/nms_batch/nms/boolean_mask_1/ShapeHwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stackJwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stack_1Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:

Kwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/Prod/reduction_indicesConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB: 
ѓ
9while/cond/batch_decode/nms_batch/nms/boolean_mask_1/ProdProdBwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_sliceKwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/Prod/reduction_indices*
T0*
_output_shapes
: 

<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Shape_1Shape1while/cond/batch_decode/nms_batch/strided_slice_1*
T0*
_output_shapes
:

Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB: *
dtype0

Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
М
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1StridedSlice<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Shape_1Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stackLwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2*
T0*
Index0*

begin_mask*
_output_shapes
: 

<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Shape_2Shape1while/cond/batch_decode/nms_batch/strided_slice_1*
T0*
_output_shapes
:

Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB: *
dtype0

Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
К
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2StridedSlice<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Shape_2Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stackLwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2*
T0*
Index0*
end_mask*
_output_shapes
: 
Е
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat/values_1Pack9while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Prod*
T0*
N*
_output_shapes
:
ю
@while/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat/axisConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 

;while/cond/batch_decode/nms_batch/nms/boolean_mask_1/concatConcatV2Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat/values_1Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2@while/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat/axis*
T0*
N*
_output_shapes
:
х
<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/ReshapeReshape1while/cond/batch_decode/nms_batch/strided_slice_1;while/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat*
T0*#
_output_shapes
:џџџџџџџџџ

Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape_1/shapeConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ь
>while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape_1Reshape-while/cond/batch_decode/nms_batch/nms/GreaterDwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape_1/shape*
T0
*#
_output_shapes
:џџџџџџџџџ
Ќ
:while/cond/batch_decode/nms_batch/nms/boolean_mask_1/WhereWhere>while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ш
<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/SqueezeSqueeze:while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Where*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ
№
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/GatherV2/axisConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
к
=while/cond/batch_decode/nms_batch/nms/boolean_mask_1/GatherV2GatherV2<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/SqueezeBwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/GatherV2/axis*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Taxis0

+while/cond/batch_decode/nms_batch/nms/WhereWhere-while/cond/batch_decode/nms_batch/nms/Greater*'
_output_shapes
:џџџџџџџџџ
ђ
3while/cond/batch_decode/nms_batch/nms/Reshape/shapeConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
Ш
-while/cond/batch_decode/nms_batch/nms/ReshapeReshape+while/cond/batch_decode/nms_batch/nms/Where3while/cond/batch_decode/nms_batch/nms/Reshape/shape*#
_output_shapes
:џџџџџџџџџ*
T0	
ј
Iwhile/cond/batch_decode/nms_batch/nms/NonMaxSuppressionV2/max_output_sizeConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value
B :Ш*
dtype0*
_output_shapes
: 
ё
9while/cond/batch_decode/nms_batch/nms/NonMaxSuppressionV2NonMaxSuppressionV2;while/cond/batch_decode/nms_batch/nms/boolean_mask/GatherV2=while/cond/batch_decode/nms_batch/nms/boolean_mask_1/GatherV2Iwhile/cond/batch_decode/nms_batch/nms/NonMaxSuppressionV2/max_output_size3while/cond/batch_decode/nms_batch/nms/iou_threshold*#
_output_shapes
:џџџџџџџџџ
с
3while/cond/batch_decode/nms_batch/nms/GatherV2/axisConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
Њ
.while/cond/batch_decode/nms_batch/nms/GatherV2GatherV2-while/cond/batch_decode/nms_batch/nms/Reshape9while/cond/batch_decode/nms_batch/nms/NonMaxSuppressionV23while/cond/batch_decode/nms_batch/nms/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*#
_output_shapes
:џџџџџџџџџ
э
7while/cond/batch_decode/nms_batch/strided_slice_2/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB: *
dtype0
я
9while/cond/batch_decode/nms_batch/strided_slice_2/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
я
9while/cond/batch_decode/nms_batch/strided_slice_2/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
ц
1while/cond/batch_decode/nms_batch/strided_slice_2StridedSlicewhile/cond/batch_decode/stack7while/cond/batch_decode/nms_batch/strided_slice_2/stack9while/cond/batch_decode/nms_batch/strided_slice_2/stack_19while/cond/batch_decode/nms_batch/strided_slice_2/stack_2*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask*
Index0*
T0
н
/while/cond/batch_decode/nms_batch/GatherV2/axisConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
_output_shapes
: *
value	B : *
dtype0

*while/cond/batch_decode/nms_batch/GatherV2GatherV21while/cond/batch_decode/nms_batch/strided_slice_2.while/cond/batch_decode/nms_batch/nms/GatherV2/while/cond/batch_decode/nms_batch/GatherV2/axis*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџ*
Taxis0
э
7while/cond/batch_decode/nms_batch/strided_slice_3/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
я
9while/cond/batch_decode/nms_batch/strided_slice_3/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
я
9while/cond/batch_decode/nms_batch/strided_slice_3/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
ц
1while/cond/batch_decode/nms_batch/strided_slice_3StridedSlice!while/cond/batch_decode/Reshape_17while/cond/batch_decode/nms_batch/strided_slice_3/stack9while/cond/batch_decode/nms_batch/strided_slice_3/stack_19while/cond/batch_decode/nms_batch/strided_slice_3/stack_2*
shrink_axis_mask*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ
п
1while/cond/batch_decode/nms_batch/GatherV2_1/axisConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 

,while/cond/batch_decode/nms_batch/GatherV2_1GatherV21while/cond/batch_decode/nms_batch/strided_slice_3.while/cond/batch_decode/nms_batch/nms/GatherV21while/cond/batch_decode/nms_batch/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ

1while/cond/batch_decode/nms_batch/batch_pad/ShapeShape*while/cond/batch_decode/nms_batch/GatherV2*
T0*
_output_shapes
:
ѕ
?while/cond/batch_decode/nms_batch/batch_pad/strided_slice/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ї
Awhile/cond/batch_decode/nms_batch/batch_pad/strided_slice/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
ї
Awhile/cond/batch_decode/nms_batch/batch_pad/strided_slice/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

9while/cond/batch_decode/nms_batch/batch_pad/strided_sliceStridedSlice1while/cond/batch_decode/nms_batch/batch_pad/Shape?while/cond/batch_decode/nms_batch/batch_pad/strided_slice/stackAwhile/cond/batch_decode/nms_batch/batch_pad/strided_slice/stack_1Awhile/cond/batch_decode/nms_batch/batch_pad/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
Ђ
1while/cond/batch_decode/nms_batch/batch_pad/stackPack9while/cond/batch_decode/nms_batch/batch_pad/strided_slice*
N*
_output_shapes
:*
T0
ч
1while/cond/batch_decode/nms_batch/batch_pad/ConstConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB: 
Н
/while/cond/batch_decode/nms_batch/batch_pad/MaxMax1while/cond/batch_decode/nms_batch/batch_pad/stack1while/cond/batch_decode/nms_batch/batch_pad/Const*
T0*
_output_shapes
: 

:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/ShapeShape*while/cond/batch_decode/nms_batch/GatherV2*
_output_shapes
:*
T0
А
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/unstackUnpack:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Shape*	
num*
T0*
_output_shapes
: : 
ш
:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
Э
8while/cond/batch_decode/nms_batch/batch_pad/pad_axis/subSub/while/cond/batch_decode/nms_batch/batch_pad/Max:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub/y*
T0*
_output_shapes
: 
к
:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub_1Sub8while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/unstack*
T0*
_output_shapes
: 
ь
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Maximum/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
ф
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/MaximumMaximum:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub_1>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Maximum/y*
T0*
_output_shapes
: 
ї
:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB"        *
dtype0*
_output_shapes
:
ь
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_1/1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
№
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_1Pack<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Maximum>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_1/1*
T0*
N*
_output_shapes
:
ќ
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_2Pack:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_1*
T0*

axis*
N*
_output_shapes

:
ф
8while/cond/batch_decode/nms_batch/batch_pad/pad_axis/PadPad*while/cond/batch_decode/nms_batch/GatherV2<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_2*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ь
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_3/1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
_output_shapes
: *
value	B :*
dtype0
у
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_3Pack/while/cond/batch_decode/nms_batch/batch_pad/Max>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_3/1*
T0*
N*
_output_shapes
:
ё
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/ReshapeReshape8while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Pad<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_3*
T0*'
_output_shapes
:џџџџџџџџџ
И
3while/cond/batch_decode/nms_batch/batch_pad/stack_1Pack<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Reshape*
T0*
N*+
_output_shapes
:џџџџџџџџџ

3while/cond/batch_decode/nms_batch/batch_pad_1/ShapeShape,while/cond/batch_decode/nms_batch/GatherV2_1*
_output_shapes
:*
T0
ї
Awhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
љ
Cwhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stack_1Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
љ
Cwhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stack_2Const3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

;while/cond/batch_decode/nms_batch/batch_pad_1/strided_sliceStridedSlice3while/cond/batch_decode/nms_batch/batch_pad_1/ShapeAwhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stackCwhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stack_1Cwhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
І
3while/cond/batch_decode/nms_batch/batch_pad_1/stackPack;while/cond/batch_decode/nms_batch/batch_pad_1/strided_slice*
T0*
N*
_output_shapes
:
щ
3while/cond/batch_decode/nms_batch/batch_pad_1/ConstConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
У
1while/cond/batch_decode/nms_batch/batch_pad_1/MaxMax3while/cond/batch_decode/nms_batch/batch_pad_1/stack3while/cond/batch_decode/nms_batch/batch_pad_1/Const*
T0*
_output_shapes
: 

<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/ShapeShape,while/cond/batch_decode/nms_batch/GatherV2_1*
T0*
_output_shapes
:
В
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/unstackUnpack<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Shape*	
num*
T0*
_output_shapes
: 
ъ
<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
value	B : 
г
:while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/subSub1while/cond/batch_decode/nms_batch/batch_pad_1/Max<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub/y*
T0*
_output_shapes
: 
р
<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub_1Sub:while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/unstack*
T0*
_output_shapes
: 
ю
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Maximum/yConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
ъ
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/MaximumMaximum<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub_1@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Maximum/y*
T0*
_output_shapes
: 
ђ
<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stackConst3^while/cond/batch_decode/assert_equal/Assert/Assert5^while/cond/batch_decode/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
Д
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_1Pack>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Maximum*
T0*
N*
_output_shapes
:

>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_2Pack<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_1*
T0*

axis*
N*
_output_shapes

:
н
:while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/PadPad,while/cond/batch_decode/nms_batch/GatherV2_1>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_2*
T0*#
_output_shapes
:џџџџџџџџџ
Ї
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_3Pack1while/cond/batch_decode/nms_batch/batch_pad_1/Max*
T0*
N*
_output_shapes
:
ѓ
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/ReshapeReshape:while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Pad>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_3*#
_output_shapes
:џџџџџџџџџ*
T0
И
5while/cond/batch_decode/nms_batch/batch_pad_1/stack_1Pack>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Reshape*
N*'
_output_shapes
:џџџџџџџџџ*
T0
v
while/cond/batch_decode_1/RankConst^while/cond/switch_t*
_output_shapes
: *
value	B :*
dtype0

(while/cond/batch_decode_1/assert_equal/yConst^while/cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
 
,while/cond/batch_decode_1/assert_equal/EqualEqualwhile/cond/batch_decode_1/Rank(while/cond/batch_decode_1/assert_equal/y*
_output_shapes
: *
T0

,while/cond/batch_decode_1/assert_equal/ConstConst^while/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
Ѕ
*while/cond/batch_decode_1/assert_equal/AllAll,while/cond/batch_decode_1/assert_equal/Equal,while/cond/batch_decode_1/assert_equal/Const*
_output_shapes
: 

3while/cond/batch_decode_1/assert_equal/Assert/ConstConst^while/cond/switch_t*
valueB B *
dtype0*
_output_shapes
: 
З
5while/cond/batch_decode_1/assert_equal/Assert/Const_1Const^while/cond/switch_t*
_output_shapes
: *<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0
Г
5while/cond/batch_decode_1/assert_equal/Assert/Const_2Const^while/cond/switch_t*8
value/B- B'x (while/cond/batch_decode_1/Rank:0) = *
dtype0*
_output_shapes
: 
Н
5while/cond/batch_decode_1/assert_equal/Assert/Const_3Const^while/cond/switch_t*B
value9B7 B1y (while/cond/batch_decode_1/assert_equal/y:0) = *
dtype0*
_output_shapes
: 

;while/cond/batch_decode_1/assert_equal/Assert/Assert/data_0Const^while/cond/switch_t*
_output_shapes
: *
valueB B *
dtype0
Н
;while/cond/batch_decode_1/assert_equal/Assert/Assert/data_1Const^while/cond/switch_t*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
Й
;while/cond/batch_decode_1/assert_equal/Assert/Assert/data_2Const^while/cond/switch_t*8
value/B- B'x (while/cond/batch_decode_1/Rank:0) = *
dtype0*
_output_shapes
: 
У
;while/cond/batch_decode_1/assert_equal/Assert/Assert/data_4Const^while/cond/switch_t*B
value9B7 B1y (while/cond/batch_decode_1/assert_equal/y:0) = *
dtype0*
_output_shapes
: 
Й
4while/cond/batch_decode_1/assert_equal/Assert/AssertAssert*while/cond/batch_decode_1/assert_equal/All;while/cond/batch_decode_1/assert_equal/Assert/Assert/data_0;while/cond/batch_decode_1/assert_equal/Assert/Assert/data_1;while/cond/batch_decode_1/assert_equal/Assert/Assert/data_2while/cond/batch_decode_1/Rank;while/cond/batch_decode_1/assert_equal/Assert/Assert/data_4(while/cond/batch_decode_1/assert_equal/y*
T

2
x
 while/cond/batch_decode_1/Rank_1Const^while/cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 

*while/cond/batch_decode_1/assert_equal_1/yConst^while/cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
І
.while/cond/batch_decode_1/assert_equal_1/EqualEqual while/cond/batch_decode_1/Rank_1*while/cond/batch_decode_1/assert_equal_1/y*
T0*
_output_shapes
: 

.while/cond/batch_decode_1/assert_equal_1/ConstConst^while/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
Ћ
,while/cond/batch_decode_1/assert_equal_1/AllAll.while/cond/batch_decode_1/assert_equal_1/Equal.while/cond/batch_decode_1/assert_equal_1/Const*
_output_shapes
: 

5while/cond/batch_decode_1/assert_equal_1/Assert/ConstConst^while/cond/switch_t*
valueB B *
dtype0*
_output_shapes
: 
Й
7while/cond/batch_decode_1/assert_equal_1/Assert/Const_1Const^while/cond/switch_t*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
З
7while/cond/batch_decode_1/assert_equal_1/Assert/Const_2Const^while/cond/switch_t*:
value1B/ B)x (while/cond/batch_decode_1/Rank_1:0) = *
dtype0*
_output_shapes
: 
С
7while/cond/batch_decode_1/assert_equal_1/Assert/Const_3Const^while/cond/switch_t*D
value;B9 B3y (while/cond/batch_decode_1/assert_equal_1/y:0) = *
dtype0*
_output_shapes
: 

=while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_0Const^while/cond/switch_t*
dtype0*
_output_shapes
: *
valueB B 
П
=while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_1Const^while/cond/switch_t*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
Н
=while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_2Const^while/cond/switch_t*:
value1B/ B)x (while/cond/batch_decode_1/Rank_1:0) = *
dtype0*
_output_shapes
: 
Ч
=while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_4Const^while/cond/switch_t*D
value;B9 B3y (while/cond/batch_decode_1/assert_equal_1/y:0) = *
dtype0*
_output_shapes
: 
Щ
6while/cond/batch_decode_1/assert_equal_1/Assert/AssertAssert,while/cond/batch_decode_1/assert_equal_1/All=while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_0=while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_1=while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_2 while/cond/batch_decode_1/Rank_1=while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_4*while/cond/batch_decode_1/assert_equal_1/y*
T

2
ь
'while/cond/batch_decode_1/Reshape/shapeConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*!
valueB"   џџџџ   *
dtype0*
_output_shapes
:
б
!while/cond/batch_decode_1/ReshapeReshapeDwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm'while/cond/batch_decode_1/Reshape/shape*
T0*+
_output_shapes
:џџџџџџџџџ
ъ
)while/cond/batch_decode_1/Reshape_1/shapeConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB"   џџџџ*
dtype0*
_output_shapes
:
З
#while/cond/batch_decode_1/Reshape_1Reshape*while/cond/PyramidFusedNet_3/strided_slice)while/cond/batch_decode_1/Reshape_1/shape*
T0*'
_output_shapes
:џџџџџџџџџ

!while/cond/batch_decode_1/unstackUnpack!while/cond/batch_decode_1/Reshape*	
num*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
Pwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/RankRankYwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch:15^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
T0*
_output_shapes
: 
я
Wwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Rank/SwitchSwitchwhile/Reshape_1while/cond/pred_id*
T0*"
_class
loc:@while/Reshape_1*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ

Qwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/sub/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 

Owhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/subSubPwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/RankQwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/sub/y*
T0*
_output_shapes
: 

Wwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range/startConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 

Wwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range/deltaConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 

Qwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/RangeRangeWwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range/startPwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/RankWwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range/delta*#
_output_shapes
:џџџџџџџџџ
Њ
Qwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/sub_1SubOwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/subQwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range*
T0*#
_output_shapes
:џџџџџџџџџ
И
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose	TransposeYwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch:1Qwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/sub_1*
T0*'
_output_shapes
:џџџџџџџџџ

Iwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstackUnpackKwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose*	
num*
T0*P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ

Ewhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/subSubKwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:3Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:1*#
_output_shapes
:џџџџџџџџџ*
T0

Gwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/sub_1SubKwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:2Iwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack*#
_output_shapes
:џџџџџџџџџ*
T0

Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 

Iwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truedivRealDivGwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/sub_1Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv/y*
T0*#
_output_shapes
:џџџџџџџџџ

Ewhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/addAddIwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstackIwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv*#
_output_shapes
:џџџџџџџџџ*
T0

Mwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv_1/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 

Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv_1RealDivEwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/subMwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv_1/y*#
_output_shapes
:џџџџџџџџџ*
T0

Gwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/add_1AddKwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:1Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ
{
/while/cond/batch_decode_1/Decode/transpose/RankRank!while/cond/batch_decode_1/unstack*
T0*
_output_shapes
: 
т
0while/cond/batch_decode_1/Decode/transpose/sub/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
Й
.while/cond/batch_decode_1/Decode/transpose/subSub/while/cond/batch_decode_1/Decode/transpose/Rank0while/cond/batch_decode_1/Decode/transpose/sub/y*
T0*
_output_shapes
: 
ш
6while/cond/batch_decode_1/Decode/transpose/Range/startConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
_output_shapes
: *
value	B : *
dtype0
ш
6while/cond/batch_decode_1/Decode/transpose/Range/deltaConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
џ
0while/cond/batch_decode_1/Decode/transpose/RangeRange6while/cond/batch_decode_1/Decode/transpose/Range/start/while/cond/batch_decode_1/Decode/transpose/Rank6while/cond/batch_decode_1/Decode/transpose/Range/delta*#
_output_shapes
:џџџџџџџџџ
Ч
0while/cond/batch_decode_1/Decode/transpose/sub_1Sub.while/cond/batch_decode_1/Decode/transpose/sub0while/cond/batch_decode_1/Decode/transpose/Range*
T0*#
_output_shapes
:џџџџџџџџџ
О
*while/cond/batch_decode_1/Decode/transpose	Transpose!while/cond/batch_decode_1/unstack0while/cond/batch_decode_1/Decode/transpose/sub_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
(while/cond/batch_decode_1/Decode/unstackUnpack*while/cond/batch_decode_1/Decode/transpose*	
num*
T0*P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ

$while/cond/batch_decode_1/Decode/ExpExp*while/cond/batch_decode_1/Decode/unstack:3*
T0*#
_output_shapes
:џџџџџџџџџ
Ц
$while/cond/batch_decode_1/Decode/mulMul$while/cond/batch_decode_1/Decode/ExpEwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/sub*
T0*#
_output_shapes
:џџџџџџџџџ

&while/cond/batch_decode_1/Decode/Exp_1Exp*while/cond/batch_decode_1/Decode/unstack:2*
T0*#
_output_shapes
:џџџџџџџџџ
Ь
&while/cond/batch_decode_1/Decode/mul_1Mul&while/cond/batch_decode_1/Decode/Exp_1Gwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/sub_1*
T0*#
_output_shapes
:џџџџџџџџџ
Ю
&while/cond/batch_decode_1/Decode/mul_2Mul(while/cond/batch_decode_1/Decode/unstackGwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/sub_1*#
_output_shapes
:џџџџџџџџџ*
T0
Ш
$while/cond/batch_decode_1/Decode/addAdd&while/cond/batch_decode_1/Decode/mul_2Ewhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/add*#
_output_shapes
:џџџџџџџџџ*
T0
Ю
&while/cond/batch_decode_1/Decode/mul_3Mul*while/cond/batch_decode_1/Decode/unstack:1Ewhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/sub*#
_output_shapes
:џџџџџџџџџ*
T0
Ь
&while/cond/batch_decode_1/Decode/add_1Add&while/cond/batch_decode_1/Decode/mul_3Gwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/add_1*#
_output_shapes
:џџџџџџџџџ*
T0
п
*while/cond/batch_decode_1/Decode/truediv/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 
Е
(while/cond/batch_decode_1/Decode/truedivRealDiv&while/cond/batch_decode_1/Decode/mul_1*while/cond/batch_decode_1/Decode/truediv/y*
T0*#
_output_shapes
:џџџџџџџџџ
Љ
$while/cond/batch_decode_1/Decode/subSub$while/cond/batch_decode_1/Decode/add(while/cond/batch_decode_1/Decode/truediv*#
_output_shapes
:џџџџџџџџџ*
T0
с
,while/cond/batch_decode_1/Decode/truediv_1/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 
З
*while/cond/batch_decode_1/Decode/truediv_1RealDiv$while/cond/batch_decode_1/Decode/mul,while/cond/batch_decode_1/Decode/truediv_1/y*#
_output_shapes
:џџџџџџџџџ*
T0
Џ
&while/cond/batch_decode_1/Decode/sub_1Sub&while/cond/batch_decode_1/Decode/add_1*while/cond/batch_decode_1/Decode/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ
с
,while/cond/batch_decode_1/Decode/truediv_2/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 
Й
*while/cond/batch_decode_1/Decode/truediv_2RealDiv&while/cond/batch_decode_1/Decode/mul_1,while/cond/batch_decode_1/Decode/truediv_2/y*
T0*#
_output_shapes
:џџџџџџџџџ
­
&while/cond/batch_decode_1/Decode/add_2Add$while/cond/batch_decode_1/Decode/add*while/cond/batch_decode_1/Decode/truediv_2*#
_output_shapes
:џџџџџџџџџ*
T0
с
,while/cond/batch_decode_1/Decode/truediv_3/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 
З
*while/cond/batch_decode_1/Decode/truediv_3RealDiv$while/cond/batch_decode_1/Decode/mul,while/cond/batch_decode_1/Decode/truediv_3/y*
T0*#
_output_shapes
:џџџџџџџџџ
Џ
&while/cond/batch_decode_1/Decode/add_3Add&while/cond/batch_decode_1/Decode/add_1*while/cond/batch_decode_1/Decode/truediv_3*#
_output_shapes
:џџџџџџџџџ*
T0

&while/cond/batch_decode_1/Decode/stackPack$while/cond/batch_decode_1/Decode/sub&while/cond/batch_decode_1/Decode/sub_1&while/cond/batch_decode_1/Decode/add_2&while/cond/batch_decode_1/Decode/add_3*'
_output_shapes
:џџџџџџџџџ*
T0*
N

1while/cond/batch_decode_1/Decode/transpose_1/RankRank&while/cond/batch_decode_1/Decode/stack*
T0*
_output_shapes
: 
ф
2while/cond/batch_decode_1/Decode/transpose_1/sub/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
П
0while/cond/batch_decode_1/Decode/transpose_1/subSub1while/cond/batch_decode_1/Decode/transpose_1/Rank2while/cond/batch_decode_1/Decode/transpose_1/sub/y*
_output_shapes
: *
T0
ъ
8while/cond/batch_decode_1/Decode/transpose_1/Range/startConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
value	B : 
ъ
8while/cond/batch_decode_1/Decode/transpose_1/Range/deltaConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 

2while/cond/batch_decode_1/Decode/transpose_1/RangeRange8while/cond/batch_decode_1/Decode/transpose_1/Range/start1while/cond/batch_decode_1/Decode/transpose_1/Rank8while/cond/batch_decode_1/Decode/transpose_1/Range/delta*#
_output_shapes
:џџџџџџџџџ
Э
2while/cond/batch_decode_1/Decode/transpose_1/sub_1Sub0while/cond/batch_decode_1/Decode/transpose_1/sub2while/cond/batch_decode_1/Decode/transpose_1/Range*
T0*#
_output_shapes
:џџџџџџџџџ
Ч
,while/cond/batch_decode_1/Decode/transpose_1	Transpose&while/cond/batch_decode_1/Decode/stack2while/cond/batch_decode_1/Decode/transpose_1/sub_1*
T0*'
_output_shapes
:џџџџџџџџџ

while/cond/batch_decode_1/stackPack,while/cond/batch_decode_1/Decode/transpose_1*
T0*
N*+
_output_shapes
:џџџџџџџџџ
ё
7while/cond/batch_decode_1/nms_batch/strided_slice/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ѓ
9while/cond/batch_decode_1/nms_batch/strided_slice/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
ѓ
9while/cond/batch_decode_1/nms_batch/strided_slice/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB:*
dtype0
ш
1while/cond/batch_decode_1/nms_batch/strided_sliceStridedSlicewhile/cond/batch_decode_1/stack7while/cond/batch_decode_1/nms_batch/strided_slice/stack9while/cond/batch_decode_1/nms_batch/strided_slice/stack_19while/cond/batch_decode_1/nms_batch/strided_slice/stack_2*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask*
T0*
Index0
ѓ
9while/cond/batch_decode_1/nms_batch/strided_slice_1/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_1/nms_batch/strided_slice_1/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_1/nms_batch/strided_slice_1/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
№
3while/cond/batch_decode_1/nms_batch/strided_slice_1StridedSlice#while/cond/batch_decode_1/Reshape_19while/cond/batch_decode_1/nms_batch/strided_slice_1/stack;while/cond/batch_decode_1/nms_batch/strided_slice_1/stack_1;while/cond/batch_decode_1/nms_batch/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*#
_output_shapes
:џџџџџџџџџ
ъ
5while/cond/batch_decode_1/nms_batch/nms/iou_thresholdConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ>
ь
7while/cond/batch_decode_1/nms_batch/nms/score_thresholdConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
ж
/while/cond/batch_decode_1/nms_batch/nms/GreaterGreater3while/cond/batch_decode_1/nms_batch/strided_slice_17while/cond/batch_decode_1/nms_batch/nms/score_threshold*
T0*#
_output_shapes
:џџџџџџџџџ

:while/cond/batch_decode_1/nms_batch/nms/boolean_mask/ShapeShape1while/cond/batch_decode_1/nms_batch/strided_slice*
_output_shapes
:*
T0

Hwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB: 

Jwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

Jwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
Ђ
Bwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_sliceStridedSlice:while/cond/batch_decode_1/nms_batch/nms/boolean_mask/ShapeHwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stackJwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stack_1Jwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stack_2*
_output_shapes
:*
T0*
Index0

Kwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/Prod/reduction_indicesConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ѓ
9while/cond/batch_decode_1/nms_batch/nms/boolean_mask/ProdProdBwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_sliceKwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/Prod/reduction_indices*
T0*
_output_shapes
: 

<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Shape_1Shape1while/cond/batch_decode_1/nms_batch/strided_slice*
T0*
_output_shapes
:

Jwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
М
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1StridedSlice<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Shape_1Jwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stackLwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stack_1Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stack_2*

begin_mask*
T0*
Index0*
_output_shapes
: 

<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Shape_2Shape1while/cond/batch_decode_1/nms_batch/strided_slice*
_output_shapes
:*
T0

Jwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:

Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB: 

Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
М
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2StridedSlice<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Shape_2Jwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stackLwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stack_1Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stack_2*
Index0*
T0*
end_mask*
_output_shapes
:
Е
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat/values_1Pack9while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Prod*
T0*
N*
_output_shapes
:
ђ
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat/axisConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
_output_shapes
: *
value	B : *
dtype0

;while/cond/batch_decode_1/nms_batch/nms/boolean_mask/concatConcatV2Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat/values_1Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2@while/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat/axis*
T0*
N*
_output_shapes
:
щ
<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/ReshapeReshape1while/cond/batch_decode_1/nms_batch/strided_slice;while/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat*
T0*'
_output_shapes
:џџџџџџџџџ

Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape_1/shapeConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ю
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape_1Reshape/while/cond/batch_decode_1/nms_batch/nms/GreaterDwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape_1/shape*
T0
*#
_output_shapes
:џџџџџџџџџ
Ќ
:while/cond/batch_decode_1/nms_batch/nms/boolean_mask/WhereWhere>while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ш
<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/SqueezeSqueeze:while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ
є
Bwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/GatherV2/axisConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
о
=while/cond/batch_decode_1/nms_batch/nms/boolean_mask/GatherV2GatherV2<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/SqueezeBwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/GatherV2/axis*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџ

<while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/ShapeShape3while/cond/batch_decode_1/nms_batch/strided_slice_1*
T0*
_output_shapes
:

Jwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB:*
dtype0
Ќ
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_sliceStridedSlice<while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/ShapeJwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stackLwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stack_1Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:

Mwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Prod/reduction_indicesConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
љ
;while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/ProdProdDwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_sliceMwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Prod/reduction_indices*
_output_shapes
: *
T0
Ё
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Shape_1Shape3while/cond/batch_decode_1/nms_batch/strided_slice_1*
T0*
_output_shapes
:

Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB: 

Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB:*
dtype0
Ц
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1StridedSlice>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Shape_1Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stackNwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2*
_output_shapes
: *
T0*
Index0*

begin_mask
Ё
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Shape_2Shape3while/cond/batch_decode_1/nms_batch/strided_slice_1*
T0*
_output_shapes
:

Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
Ф
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2StridedSlice>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Shape_2Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stackNwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2*
_output_shapes
: *
Index0*
T0*
end_mask
Й
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat/values_1Pack;while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Prod*
T0*
N*
_output_shapes
:
є
Bwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat/axisConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 

=while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concatConcatV2Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat/values_1Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2Bwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat/axis*
T0*
N*
_output_shapes
:
ы
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/ReshapeReshape3while/cond/batch_decode_1/nms_batch/strided_slice_1=while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat*
T0*#
_output_shapes
:џџџџџџџџџ

Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape_1/shapeConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
ђ
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape_1Reshape/while/cond/batch_decode_1/nms_batch/nms/GreaterFwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape_1/shape*#
_output_shapes
:џџџџџџџџџ*
T0

А
<while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/WhereWhere@while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ь
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/SqueezeSqueeze<while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Where*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ
і
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/GatherV2/axisConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
т
?while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/GatherV2GatherV2>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/SqueezeDwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/GatherV2/axis*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Taxis0

-while/cond/batch_decode_1/nms_batch/nms/WhereWhere/while/cond/batch_decode_1/nms_batch/nms/Greater*'
_output_shapes
:џџџџџџџџџ
ј
5while/cond/batch_decode_1/nms_batch/nms/Reshape/shapeConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
Ю
/while/cond/batch_decode_1/nms_batch/nms/ReshapeReshape-while/cond/batch_decode_1/nms_batch/nms/Where5while/cond/batch_decode_1/nms_batch/nms/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ
ў
Kwhile/cond/batch_decode_1/nms_batch/nms/NonMaxSuppressionV2/max_output_sizeConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
_output_shapes
: *
value
B :Ш*
dtype0
ћ
;while/cond/batch_decode_1/nms_batch/nms/NonMaxSuppressionV2NonMaxSuppressionV2=while/cond/batch_decode_1/nms_batch/nms/boolean_mask/GatherV2?while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/GatherV2Kwhile/cond/batch_decode_1/nms_batch/nms/NonMaxSuppressionV2/max_output_size5while/cond/batch_decode_1/nms_batch/nms/iou_threshold*#
_output_shapes
:џџџџџџџџџ
ч
5while/cond/batch_decode_1/nms_batch/nms/GatherV2/axisConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
_output_shapes
: *
value	B : *
dtype0
В
0while/cond/batch_decode_1/nms_batch/nms/GatherV2GatherV2/while/cond/batch_decode_1/nms_batch/nms/Reshape;while/cond/batch_decode_1/nms_batch/nms/NonMaxSuppressionV25while/cond/batch_decode_1/nms_batch/nms/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*#
_output_shapes
:џџџџџџџџџ
ѓ
9while/cond/batch_decode_1/nms_batch/strided_slice_2/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_1/nms_batch/strided_slice_2/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_1/nms_batch/strided_slice_2/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
№
3while/cond/batch_decode_1/nms_batch/strided_slice_2StridedSlicewhile/cond/batch_decode_1/stack9while/cond/batch_decode_1/nms_batch/strided_slice_2/stack;while/cond/batch_decode_1/nms_batch/strided_slice_2/stack_1;while/cond/batch_decode_1/nms_batch/strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*'
_output_shapes
:џџџџџџџџџ
у
1while/cond/batch_decode_1/nms_batch/GatherV2/axisConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
Ї
,while/cond/batch_decode_1/nms_batch/GatherV2GatherV23while/cond/batch_decode_1/nms_batch/strided_slice_20while/cond/batch_decode_1/nms_batch/nms/GatherV21while/cond/batch_decode_1/nms_batch/GatherV2/axis*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџ*
Taxis0
ѓ
9while/cond/batch_decode_1/nms_batch/strided_slice_3/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_1/nms_batch/strided_slice_3/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_1/nms_batch/strided_slice_3/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
№
3while/cond/batch_decode_1/nms_batch/strided_slice_3StridedSlice#while/cond/batch_decode_1/Reshape_19while/cond/batch_decode_1/nms_batch/strided_slice_3/stack;while/cond/batch_decode_1/nms_batch/strided_slice_3/stack_1;while/cond/batch_decode_1/nms_batch/strided_slice_3/stack_2*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask*
Index0*
T0
х
3while/cond/batch_decode_1/nms_batch/GatherV2_1/axisConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
Ї
.while/cond/batch_decode_1/nms_batch/GatherV2_1GatherV23while/cond/batch_decode_1/nms_batch/strided_slice_30while/cond/batch_decode_1/nms_batch/nms/GatherV23while/cond/batch_decode_1/nms_batch/GatherV2_1/axis*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Taxis0

3while/cond/batch_decode_1/nms_batch/batch_pad/ShapeShape,while/cond/batch_decode_1/nms_batch/GatherV2*
T0*
_output_shapes
:
ћ
Awhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
§
Cwhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
§
Cwhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

;while/cond/batch_decode_1/nms_batch/batch_pad/strided_sliceStridedSlice3while/cond/batch_decode_1/nms_batch/batch_pad/ShapeAwhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stackCwhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stack_1Cwhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
І
3while/cond/batch_decode_1/nms_batch/batch_pad/stackPack;while/cond/batch_decode_1/nms_batch/batch_pad/strided_slice*
T0*
N*
_output_shapes
:
э
3while/cond/batch_decode_1/nms_batch/batch_pad/ConstConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
У
1while/cond/batch_decode_1/nms_batch/batch_pad/MaxMax3while/cond/batch_decode_1/nms_batch/batch_pad/stack3while/cond/batch_decode_1/nms_batch/batch_pad/Const*
T0*
_output_shapes
: 

<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/ShapeShape,while/cond/batch_decode_1/nms_batch/GatherV2*
T0*
_output_shapes
:
Д
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/unstackUnpack<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Shape*
_output_shapes
: : *	
num*
T0
ю
<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
г
:while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/subSub1while/cond/batch_decode_1/nms_batch/batch_pad/Max<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub/y*
T0*
_output_shapes
: 
р
<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub_1Sub:while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/unstack*
T0*
_output_shapes
: 
ђ
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Maximum/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
ъ
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/MaximumMaximum<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub_1@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Maximum/y*
_output_shapes
: *
T0
§
<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB"        *
dtype0*
_output_shapes
:
ђ
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_1/1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
і
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_1Pack>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Maximum@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_1/1*
N*
_output_shapes
:*
T0

>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_2Pack<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_1*
T0*

axis*
N*
_output_shapes

:
ъ
:while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/PadPad,while/cond/batch_decode_1/nms_batch/GatherV2>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_2*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
ђ
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_3/1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
щ
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_3Pack1while/cond/batch_decode_1/nms_batch/batch_pad/Max@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_3/1*
T0*
N*
_output_shapes
:
ї
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/ReshapeReshape:while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Pad>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_3*'
_output_shapes
:џџџџџџџџџ*
T0
М
5while/cond/batch_decode_1/nms_batch/batch_pad/stack_1Pack>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Reshape*
N*+
_output_shapes
:џџџџџџџџџ*
T0

5while/cond/batch_decode_1/nms_batch/batch_pad_1/ShapeShape.while/cond/batch_decode_1/nms_batch/GatherV2_1*
_output_shapes
:*
T0
§
Cwhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
џ
Ewhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stack_1Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
џ
Ewhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stack_2Const5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

=while/cond/batch_decode_1/nms_batch/batch_pad_1/strided_sliceStridedSlice5while/cond/batch_decode_1/nms_batch/batch_pad_1/ShapeCwhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stackEwhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stack_1Ewhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
Њ
5while/cond/batch_decode_1/nms_batch/batch_pad_1/stackPack=while/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice*
N*
_output_shapes
:*
T0
я
5while/cond/batch_decode_1/nms_batch/batch_pad_1/ConstConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB: *
dtype0
Щ
3while/cond/batch_decode_1/nms_batch/batch_pad_1/MaxMax5while/cond/batch_decode_1/nms_batch/batch_pad_1/stack5while/cond/batch_decode_1/nms_batch/batch_pad_1/Const*
T0*
_output_shapes
: 

>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/ShapeShape.while/cond/batch_decode_1/nms_batch/GatherV2_1*
T0*
_output_shapes
:
Ж
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/unstackUnpack>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Shape*	
num*
T0*
_output_shapes
: 
№
>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
value	B : 
й
<while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/subSub3while/cond/batch_decode_1/nms_batch/batch_pad_1/Max>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub/y*
T0*
_output_shapes
: 
ц
>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub_1Sub<while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/unstack*
T0*
_output_shapes
: 
є
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Maximum/yConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
№
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/MaximumMaximum>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub_1Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Maximum/y*
_output_shapes
: *
T0
ј
>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stackConst5^while/cond/batch_decode_1/assert_equal/Assert/Assert7^while/cond/batch_decode_1/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
И
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_1Pack@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Maximum*
N*
_output_shapes
:*
T0

@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_2Pack>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_1*
T0*

axis*
N*
_output_shapes

:
у
<while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/PadPad.while/cond/batch_decode_1/nms_batch/GatherV2_1@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_2*
T0*#
_output_shapes
:џџџџџџџџџ
Ћ
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_3Pack3while/cond/batch_decode_1/nms_batch/batch_pad_1/Max*
T0*
N*
_output_shapes
:
љ
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/ReshapeReshape<while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Pad@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_3*
T0*#
_output_shapes
:џџџџџџџџџ
М
7while/cond/batch_decode_1/nms_batch/batch_pad_1/stack_1Pack@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Reshape*
T0*
N*'
_output_shapes
:џџџџџџџџџ
p
while/cond/concat_1/axisConst^while/cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ф
while/cond/concat_1ConcatV23while/cond/batch_decode/nms_batch/batch_pad/stack_15while/cond/batch_decode_1/nms_batch/batch_pad/stack_1while/cond/concat_1/axis*
N*+
_output_shapes
:џџџџџџџџџ*
T0
p
while/cond/concat_2/axisConst^while/cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ф
while/cond/concat_2ConcatV25while/cond/batch_decode/nms_batch/batch_pad_1/stack_17while/cond/batch_decode_1/nms_batch/batch_pad_1/stack_1while/cond/concat_2/axis*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Ќ
=while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/ShapeConst^while/cond/switch_f*%
valueB"             *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/dilation_rateConst^while/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"      
л
Awhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwiseDepthwiseConv2dNativeHwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise/SwitchJwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise/Switch_1*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`*
paddingSAME

Hwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise/SwitchSwitchwhile/TensorArrayReadV3while/cond/pred_id*
T0**
_class 
loc:@while/TensorArrayReadV3*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ
Ь
Jwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enterwhile/cond/pred_id*8
_output_shapes&
$: : *
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enter
А
7while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise>while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/Switch*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
Ќ
>while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/SwitchSwitch;while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enterwhile/cond/pred_id*
T0*N
_classD
B@loc:@while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enter*8
_output_shapes&
$:` :` 
А
?while/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2dFwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/SwitchHwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_1Hwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_2Hwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_3*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ : : : : *
is_training( *
epsilon%o:
Ќ
Fwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/SwitchSwitchCwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*V
_classL
JHloc:@while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
: : 
В
Hwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
: : 
В
Hwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_2SwitchEwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
: : 
В
Hwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_3SwitchEwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id* 
_output_shapes
: : *
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3

6while/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_4/fem_conv0/LeakyRelu/alphaConst^while/cond/switch_f*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_4/fem_conv0/LeakyRelu/mulMul6while/cond/PyramidFusedNet_4/fem_conv0/LeakyRelu/alpha?while/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ *
T0
ѕ
0while/cond/PyramidFusedNet_4/fem_conv0/LeakyReluMaximum4while/cond/PyramidFusedNet_4/fem_conv0/LeakyRelu/mul?while/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
Ќ
=while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/ShapeConst^while/cond/switch_f*
dtype0*
_output_shapes
:*%
valueB"             
Ќ
Ewhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/dilation_rateConst^while/cond/switch_f*
_output_shapes
:*
valueB"      *
dtype0
С
Awhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_4/fem_conv0/LeakyReluHwhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/depthwise/Switch*
paddingSAME*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`
Ъ
Hwhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/depthwise/SwitchSwitchEwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enterwhile/cond/pred_id*8
_output_shapes&
$: : *
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enter
А
7while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/depthwise>while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/Switch*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
Ќ
>while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/SwitchSwitch;while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enterwhile/cond/pred_id*
T0*N
_classD
B@loc:@while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enter*8
_output_shapes&
$:`@:`@
А
?while/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2dFwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/SwitchHwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_1Hwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_2Hwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_3*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( *
epsilon%o:*
T0
Ќ
Fwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/SwitchSwitchCwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*V
_classL
JHloc:@while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
:@:@
В
Hwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
:@:@
В
Hwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_2SwitchEwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
:@:@*
T0
В
Hwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_3SwitchEwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id* 
_output_shapes
:@:@*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3

6while/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_4/fem_conv1/LeakyRelu/alphaConst^while/cond/switch_f*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
ї
4while/cond/PyramidFusedNet_4/fem_conv1/LeakyRelu/mulMul6while/cond/PyramidFusedNet_4/fem_conv1/LeakyRelu/alpha?while/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
ѕ
0while/cond/PyramidFusedNet_4/fem_conv1/LeakyReluMaximum4while/cond/PyramidFusedNet_4/fem_conv1/LeakyRelu/mul?while/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
Ќ
=while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/ShapeConst^while/cond/switch_f*
_output_shapes
:*%
valueB"      @      *
dtype0
Ќ
Ewhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/dilation_rateConst^while/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"      
Т
Awhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_4/fem_conv1/LeakyReluHwhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/depthwise/Switch*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР*
paddingSAME
Ъ
Hwhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/depthwise/SwitchSwitchEwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enter*8
_output_shapes&
$:@:@
А
7while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/depthwise>while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/Switch*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
Ў
>while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/SwitchSwitch;while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enterwhile/cond/pred_id*
T0*N
_classD
B@loc:@while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enter*:
_output_shapes(
&:Р@:Р@
А
?while/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2dFwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/SwitchHwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_1Hwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_2Hwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_3*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( *
epsilon%o:
Ќ
Fwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/SwitchSwitchCwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*V
_classL
JHloc:@while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
:@:@*
T0
В
Hwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
:@:@
В
Hwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_2SwitchEwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
:@:@
В
Hwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_3SwitchEwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
:@:@

6while/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_4/fem_conv2/LeakyRelu/alphaConst^while/cond/switch_f*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_4/fem_conv2/LeakyRelu/mulMul6while/cond/PyramidFusedNet_4/fem_conv2/LeakyRelu/alpha?while/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
ѕ
0while/cond/PyramidFusedNet_4/fem_conv2/LeakyReluMaximum4while/cond/PyramidFusedNet_4/fem_conv2/LeakyRelu/mul?while/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
Ќ
=while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/ShapeConst^while/cond/switch_f*%
valueB"      @      *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/dilation_rateConst^while/cond/switch_f*
valueB"      *
dtype0*
_output_shapes
:
Т
Awhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_4/fem_conv2/LeakyReluHwhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/depthwise/Switch*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР*
paddingSAME
Ъ
Hwhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/depthwise/SwitchSwitchEwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enter*8
_output_shapes&
$:@:@
А
7while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/depthwise>while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/Switch*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
Ў
>while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/SwitchSwitch;while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enterwhile/cond/pred_id*:
_output_shapes(
&:Р@:Р@*
T0*N
_classD
B@loc:@while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enter
А
?while/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2dFwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/SwitchHwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_1Hwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_2Hwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_3*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( *
epsilon%o:
Ќ
Fwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/SwitchSwitchCwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*V
_classL
JHloc:@while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
:@:@
В
Hwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
:@:@*
T0
В
Hwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_2SwitchEwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id* 
_output_shapes
:@:@*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2
В
Hwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_3SwitchEwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
:@:@

6while/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_4/fem_conv3/LeakyRelu/alphaConst^while/cond/switch_f*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_4/fem_conv3/LeakyRelu/mulMul6while/cond/PyramidFusedNet_4/fem_conv3/LeakyRelu/alpha?while/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
ѕ
0while/cond/PyramidFusedNet_4/fem_conv3/LeakyReluMaximum4while/cond/PyramidFusedNet_4/fem_conv3/LeakyRelu/mul?while/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
Ќ
=while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/ShapeConst^while/cond/switch_f*%
valueB"      @      *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/dilation_rateConst^while/cond/switch_f*
valueB"      *
dtype0*
_output_shapes
:
Т
Awhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_4/fem_conv3/LeakyReluHwhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/depthwise/Switch*
paddingSAME*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР
Ъ
Hwhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/depthwise/SwitchSwitchEwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enterwhile/cond/pred_id*8
_output_shapes&
$:@:@*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enter
А
7while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/depthwise>while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/Switch*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
Ў
>while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/SwitchSwitch;while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enterwhile/cond/pred_id*:
_output_shapes(
&:Р@:Р@*
T0*N
_classD
B@loc:@while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enter
А
?while/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2dFwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/SwitchHwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_1Hwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_2Hwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_3*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( *
epsilon%o:
Ќ
Fwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/SwitchSwitchCwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*V
_classL
JHloc:@while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
:@:@
В
Hwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_1SwitchEwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id* 
_output_shapes
:@:@*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1
В
Hwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_2SwitchEwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
:@:@
В
Hwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_3SwitchEwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*X
_classN
LJloc:@while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
:@:@

6while/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu/alphaConst^while/cond/switch_f*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu/mulMul6while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu/alpha?while/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
ѕ
0while/cond/PyramidFusedNet_4/fem_conv4/LeakyReluMaximum4while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu/mul?while/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
Ќ
=while/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/ShapeConst^while/cond/switch_f*%
valueB"             *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/dilation_rateConst^while/cond/switch_f*
valueB"      *
dtype0*
_output_shapes
:
л
Awhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/depthwiseDepthwiseConv2dNativeHwhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/depthwise/SwitchJwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise/Switch_1*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`*
paddingSAME*
T0*
strides


Hwhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/depthwise/SwitchSwitchwhile/ResizeBilinearwhile/cond/pred_id*
T0*'
_class
loc:@while/ResizeBilinear*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ
А
7while/cond/PyramidFusedNet_5/fem_conv0/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/depthwise>while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/Switch*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ *
paddingVALID
А
?while/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_5/fem_conv0/separable_conv2dFwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/SwitchHwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_1Hwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_2Hwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_3*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ : : : : *
is_training( *
epsilon%o:*
T0

6while/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_5/fem_conv0/LeakyRelu/alphaConst^while/cond/switch_f*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_5/fem_conv0/LeakyRelu/mulMul6while/cond/PyramidFusedNet_5/fem_conv0/LeakyRelu/alpha?while/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ *
T0
ѕ
0while/cond/PyramidFusedNet_5/fem_conv0/LeakyReluMaximum4while/cond/PyramidFusedNet_5/fem_conv0/LeakyRelu/mul?while/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
Ќ
=while/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d/ShapeConst^while/cond/switch_f*%
valueB"             *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d/dilation_rateConst^while/cond/switch_f*
valueB"      *
dtype0*
_output_shapes
:
С
Awhile/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_5/fem_conv0/LeakyReluHwhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/depthwise/Switch*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`*
paddingSAME*
T0
А
7while/cond/PyramidFusedNet_5/fem_conv1/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d/depthwise>while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/Switch*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
А
?while/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_5/fem_conv1/separable_conv2dFwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/SwitchHwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_1Hwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_2Hwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_3*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( 

6while/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/ConstConst^while/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *wО?

6while/cond/PyramidFusedNet_5/fem_conv1/LeakyRelu/alphaConst^while/cond/switch_f*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_5/fem_conv1/LeakyRelu/mulMul6while/cond/PyramidFusedNet_5/fem_conv1/LeakyRelu/alpha?while/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
ѕ
0while/cond/PyramidFusedNet_5/fem_conv1/LeakyReluMaximum4while/cond/PyramidFusedNet_5/fem_conv1/LeakyRelu/mul?while/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
Ќ
=while/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d/ShapeConst^while/cond/switch_f*%
valueB"      @      *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d/dilation_rateConst^while/cond/switch_f*
valueB"      *
dtype0*
_output_shapes
:
Т
Awhile/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_5/fem_conv1/LeakyReluHwhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/depthwise/Switch*
paddingSAME*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР
А
7while/cond/PyramidFusedNet_5/fem_conv2/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d/depthwise>while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/Switch*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
А
?while/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_5/fem_conv2/separable_conv2dFwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/SwitchHwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_1Hwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_2Hwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_3*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( 

6while/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_5/fem_conv2/LeakyRelu/alphaConst^while/cond/switch_f*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_5/fem_conv2/LeakyRelu/mulMul6while/cond/PyramidFusedNet_5/fem_conv2/LeakyRelu/alpha?while/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
ѕ
0while/cond/PyramidFusedNet_5/fem_conv2/LeakyReluMaximum4while/cond/PyramidFusedNet_5/fem_conv2/LeakyRelu/mul?while/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
Ќ
=while/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d/ShapeConst^while/cond/switch_f*
dtype0*
_output_shapes
:*%
valueB"      @      
Ќ
Ewhile/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d/dilation_rateConst^while/cond/switch_f*
valueB"      *
dtype0*
_output_shapes
:
Т
Awhile/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_5/fem_conv2/LeakyReluHwhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/depthwise/Switch*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР*
paddingSAME*
T0
А
7while/cond/PyramidFusedNet_5/fem_conv3/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d/depthwise>while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/Switch*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
paddingVALID*
T0
А
?while/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_5/fem_conv3/separable_conv2dFwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/SwitchHwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_1Hwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_2Hwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_3*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( *
epsilon%o:

6while/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_5/fem_conv3/LeakyRelu/alphaConst^while/cond/switch_f*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_5/fem_conv3/LeakyRelu/mulMul6while/cond/PyramidFusedNet_5/fem_conv3/LeakyRelu/alpha?while/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
ѕ
0while/cond/PyramidFusedNet_5/fem_conv3/LeakyReluMaximum4while/cond/PyramidFusedNet_5/fem_conv3/LeakyRelu/mul?while/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
T0
Ќ
=while/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d/ShapeConst^while/cond/switch_f*%
valueB"      @      *
dtype0*
_output_shapes
:
Ќ
Ewhile/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d/dilation_rateConst^while/cond/switch_f*
valueB"      *
dtype0*
_output_shapes
:
Т
Awhile/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d/depthwiseDepthwiseConv2dNative0while/cond/PyramidFusedNet_5/fem_conv3/LeakyReluHwhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/depthwise/Switch*
paddingSAME*
T0*
strides
*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџР
А
7while/cond/PyramidFusedNet_5/fem_conv4/separable_conv2dConv2DAwhile/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d/depthwise>while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/Switch*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
paddingVALID*
T0
А
?while/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNormFusedBatchNorm7while/cond/PyramidFusedNet_5/fem_conv4/separable_conv2dFwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/SwitchHwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_1Hwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_2Hwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_3*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ@:@:@:@:@*
is_training( 

6while/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

6while/cond/PyramidFusedNet_5/fem_conv4/LeakyRelu/alphaConst^while/cond/switch_f*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
ї
4while/cond/PyramidFusedNet_5/fem_conv4/LeakyRelu/mulMul6while/cond/PyramidFusedNet_5/fem_conv4/LeakyRelu/alpha?while/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
ѕ
0while/cond/PyramidFusedNet_5/fem_conv4/LeakyReluMaximum4while/cond/PyramidFusedNet_5/fem_conv4/LeakyRelu/mul?while/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
y
while/cond/concat_3/axisConst^while/cond/switch_f*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
ъ
while/cond/concat_3ConcatV20while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu0while/cond/PyramidFusedNet_5/fem_conv3/LeakyReluwhile/cond/concat_3/axis*
N*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџ*
T0
Б
Bwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/ShapeConst^while/cond/switch_f*
_output_shapes
:*%
valueB"            *
dtype0
Б
Jwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/dilation_rateConst^while/cond/switch_f*
_output_shapes
:*
valueB"      *
dtype0
Џ
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/depthwiseDepthwiseConv2dNativewhile/cond/concat_3Mwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/depthwise/Switch*9
_output_shapes'
%:#џџџџџџџџџџџџџџџџџџ*
paddingSAME*
T0*
strides

п
Mwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enterwhile/cond/pred_id*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enter*:
_output_shapes(
&::*
T0
П
<while/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/depthwiseCwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/Switch*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ *
paddingVALID*
T0*
strides

С
Cwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enterwhile/cond/pred_id*
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enter*:
_output_shapes(
&: : 
Ю
Dwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2dKwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/SwitchMwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_1Mwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_2Mwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_3*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ : : : : *
is_training( *
epsilon%o:
П
Kwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
: : 
Х
Mwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
: : 
Х
Mwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id* 
_output_shapes
: : *
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2
Х
Mwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
: : 

;while/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

;while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyRelu/alphaConst^while/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>

9while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyRelu/mulMul;while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyRelu/alphaDwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 

5while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyReluMaximum9while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyRelu/mulDwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 
Б
Bwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/ShapeConst^while/cond/switch_f*%
valueB"             *
dtype0*
_output_shapes
:
Б
Jwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/dilation_rateConst^while/cond/switch_f*
valueB"      *
dtype0*
_output_shapes
:
а
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/depthwiseDepthwiseConv2dNative5while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyReluMwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/depthwise/Switch*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`*
paddingSAME*
T0
н
Mwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enterwhile/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enter*8
_output_shapes&
$: : 
П
<while/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/depthwiseCwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/Switch*
paddingVALID*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ
П
Cwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enterwhile/cond/pred_id*
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enter*8
_output_shapes&
$:`:`
Ю
Dwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2dKwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/SwitchMwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_1Mwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_2Mwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_3*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ::::*
is_training( 
П
Kwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id* 
_output_shapes
::*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter
Х
Mwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1* 
_output_shapes
::
Х
Mwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
::
Х
Mwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
::

;while/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

2while/cond/PyramidFusedNet_6/softmax/Reshape/shapeConst^while/cond/switch_f*
valueB"џџџџ   *
dtype0*
_output_shapes
:
у
,while/cond/PyramidFusedNet_6/softmax/ReshapeReshapeDwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm2while/cond/PyramidFusedNet_6/softmax/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

,while/cond/PyramidFusedNet_6/softmax/SoftmaxSoftmax,while/cond/PyramidFusedNet_6/softmax/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ

*while/cond/PyramidFusedNet_6/softmax/ShapeShapeDwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm*
T0*
_output_shapes
:
ж
.while/cond/PyramidFusedNet_6/softmax/Reshape_1Reshape,while/cond/PyramidFusedNet_6/softmax/Softmax*while/cond/PyramidFusedNet_6/softmax/Shape*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

0while/cond/PyramidFusedNet_6/strided_slice/stackConst^while/cond/switch_f*
_output_shapes
:*%
valueB"               *
dtype0
Ё
2while/cond/PyramidFusedNet_6/strided_slice/stack_1Const^while/cond/switch_f*%
valueB"               *
dtype0*
_output_shapes
:
Ё
2while/cond/PyramidFusedNet_6/strided_slice/stack_2Const^while/cond/switch_f*%
valueB"            *
dtype0*
_output_shapes
:

*while/cond/PyramidFusedNet_6/strided_sliceStridedSlice.while/cond/PyramidFusedNet_6/softmax/Reshape_10while/cond/PyramidFusedNet_6/strided_slice/stack2while/cond/PyramidFusedNet_6/strided_slice/stack_12while/cond/PyramidFusedNet_6/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
end_mask*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Б
Bwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/ShapeConst^while/cond/switch_f*%
valueB"             *
dtype0*
_output_shapes
:
Б
Jwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/dilation_rateConst^while/cond/switch_f*
valueB"      *
dtype0*
_output_shapes
:
а
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/depthwiseDepthwiseConv2dNative5while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyReluMwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/depthwise/Switch*
paddingSAME*
T0*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`
н
Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enterwhile/cond/pred_id*8
_output_shapes&
$: : *
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enter
П
<while/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/depthwiseCwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/Switch*
strides
*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ *
paddingVALID*
T0
П
Cwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enterwhile/cond/pred_id*
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enter*8
_output_shapes&
$:` :` 
Ю
Dwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2dKwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/SwitchMwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ : : : : *
is_training( *
epsilon%o:
П
Kwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
: : 
Х
Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id* 
_output_shapes
: : *
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1
Х
Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
: : 
Х
Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
: : 

;while/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/ConstConst^while/cond/switch_f*
valueB
 *wО?*
dtype0*
_output_shapes
: 

;while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyRelu/alphaConst^while/cond/switch_f*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 

9while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyRelu/mulMul;while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyRelu/alphaDwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ 

5while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyReluMaximum9while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyRelu/mulDwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ *
T0
Б
Bwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/ShapeConst^while/cond/switch_f*%
valueB"             *
dtype0*
_output_shapes
:
Б
Jwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/dilation_rateConst^while/cond/switch_f*
valueB"      *
dtype0*
_output_shapes
:
а
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/depthwiseDepthwiseConv2dNative5while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyReluMwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/depthwise/Switch*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ`*
paddingSAME*
T0*
strides

н
Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/depthwise/SwitchSwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enterwhile/cond/pred_id*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enter*8
_output_shapes&
$: : *
T0
П
<while/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2dConv2DFwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/depthwiseCwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/Switch*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
paddingVALID*
T0*
strides

П
Cwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/SwitchSwitchBwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enterwhile/cond/pred_id*8
_output_shapes&
$:`:`*
T0*U
_classK
IGloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enter
Ю
Dwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNormFusedBatchNorm<while/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2dKwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/SwitchMwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3*
epsilon%o:*
T0*P
_output_shapes>
<:"џџџџџџџџџџџџџџџџџџ::::*
is_training( 
П
Kwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/SwitchSwitchJwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enterwhile/cond/pred_id*
T0*]
_classS
QOloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter* 
_output_shapes
::
Х
Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1while/cond/pred_id* 
_output_shapes
::*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1
Х
Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2* 
_output_shapes
::
Х
Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3SwitchLwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3while/cond/pred_id*
T0*_
_classU
SQloc:@while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3* 
_output_shapes
::

;while/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/ConstConst^while/cond/switch_f*
_output_shapes
: *
valueB
 *wО?*
dtype0
v
while/cond/batch_decode_2/RankConst^while/cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 

(while/cond/batch_decode_2/assert_equal/yConst^while/cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
 
,while/cond/batch_decode_2/assert_equal/EqualEqualwhile/cond/batch_decode_2/Rank(while/cond/batch_decode_2/assert_equal/y*
T0*
_output_shapes
: 

,while/cond/batch_decode_2/assert_equal/ConstConst^while/cond/switch_f*
_output_shapes
: *
valueB *
dtype0
Ѕ
*while/cond/batch_decode_2/assert_equal/AllAll,while/cond/batch_decode_2/assert_equal/Equal,while/cond/batch_decode_2/assert_equal/Const*
_output_shapes
: 

3while/cond/batch_decode_2/assert_equal/Assert/ConstConst^while/cond/switch_f*
valueB B *
dtype0*
_output_shapes
: 
З
5while/cond/batch_decode_2/assert_equal/Assert/Const_1Const^while/cond/switch_f*
_output_shapes
: *<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0
Г
5while/cond/batch_decode_2/assert_equal/Assert/Const_2Const^while/cond/switch_f*8
value/B- B'x (while/cond/batch_decode_2/Rank:0) = *
dtype0*
_output_shapes
: 
Н
5while/cond/batch_decode_2/assert_equal/Assert/Const_3Const^while/cond/switch_f*
_output_shapes
: *B
value9B7 B1y (while/cond/batch_decode_2/assert_equal/y:0) = *
dtype0

;while/cond/batch_decode_2/assert_equal/Assert/Assert/data_0Const^while/cond/switch_f*
valueB B *
dtype0*
_output_shapes
: 
Н
;while/cond/batch_decode_2/assert_equal/Assert/Assert/data_1Const^while/cond/switch_f*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
Й
;while/cond/batch_decode_2/assert_equal/Assert/Assert/data_2Const^while/cond/switch_f*8
value/B- B'x (while/cond/batch_decode_2/Rank:0) = *
dtype0*
_output_shapes
: 
У
;while/cond/batch_decode_2/assert_equal/Assert/Assert/data_4Const^while/cond/switch_f*B
value9B7 B1y (while/cond/batch_decode_2/assert_equal/y:0) = *
dtype0*
_output_shapes
: 
Й
4while/cond/batch_decode_2/assert_equal/Assert/AssertAssert*while/cond/batch_decode_2/assert_equal/All;while/cond/batch_decode_2/assert_equal/Assert/Assert/data_0;while/cond/batch_decode_2/assert_equal/Assert/Assert/data_1;while/cond/batch_decode_2/assert_equal/Assert/Assert/data_2while/cond/batch_decode_2/Rank;while/cond/batch_decode_2/assert_equal/Assert/Assert/data_4(while/cond/batch_decode_2/assert_equal/y*
T

2
x
 while/cond/batch_decode_2/Rank_1Const^while/cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 

*while/cond/batch_decode_2/assert_equal_1/yConst^while/cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
І
.while/cond/batch_decode_2/assert_equal_1/EqualEqual while/cond/batch_decode_2/Rank_1*while/cond/batch_decode_2/assert_equal_1/y*
_output_shapes
: *
T0

.while/cond/batch_decode_2/assert_equal_1/ConstConst^while/cond/switch_f*
valueB *
dtype0*
_output_shapes
: 
Ћ
,while/cond/batch_decode_2/assert_equal_1/AllAll.while/cond/batch_decode_2/assert_equal_1/Equal.while/cond/batch_decode_2/assert_equal_1/Const*
_output_shapes
: 

5while/cond/batch_decode_2/assert_equal_1/Assert/ConstConst^while/cond/switch_f*
_output_shapes
: *
valueB B *
dtype0
Й
7while/cond/batch_decode_2/assert_equal_1/Assert/Const_1Const^while/cond/switch_f*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
З
7while/cond/batch_decode_2/assert_equal_1/Assert/Const_2Const^while/cond/switch_f*
_output_shapes
: *:
value1B/ B)x (while/cond/batch_decode_2/Rank_1:0) = *
dtype0
С
7while/cond/batch_decode_2/assert_equal_1/Assert/Const_3Const^while/cond/switch_f*D
value;B9 B3y (while/cond/batch_decode_2/assert_equal_1/y:0) = *
dtype0*
_output_shapes
: 

=while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_0Const^while/cond/switch_f*
dtype0*
_output_shapes
: *
valueB B 
П
=while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_1Const^while/cond/switch_f*
_output_shapes
: *<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0
Н
=while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_2Const^while/cond/switch_f*:
value1B/ B)x (while/cond/batch_decode_2/Rank_1:0) = *
dtype0*
_output_shapes
: 
Ч
=while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_4Const^while/cond/switch_f*D
value;B9 B3y (while/cond/batch_decode_2/assert_equal_1/y:0) = *
dtype0*
_output_shapes
: 
Щ
6while/cond/batch_decode_2/assert_equal_1/Assert/AssertAssert,while/cond/batch_decode_2/assert_equal_1/All=while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_0=while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_1=while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_2 while/cond/batch_decode_2/Rank_1=while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_4*while/cond/batch_decode_2/assert_equal_1/y*
T

2
ь
'while/cond/batch_decode_2/Reshape/shapeConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*!
valueB"   џџџџ   
б
!while/cond/batch_decode_2/ReshapeReshapeDwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm'while/cond/batch_decode_2/Reshape/shape*
T0*+
_output_shapes
:џџџџџџџџџ
ъ
)while/cond/batch_decode_2/Reshape_1/shapeConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB"   џџџџ
З
#while/cond/batch_decode_2/Reshape_1Reshape*while/cond/PyramidFusedNet_6/strided_slice)while/cond/batch_decode_2/Reshape_1/shape*
T0*'
_output_shapes
:џџџџџџџџџ

!while/cond/batch_decode_2/unstackUnpack!while/cond/batch_decode_2/Reshape*	
num*
T0*'
_output_shapes
:џџџџџџџџџ
Т
Pwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/RankRankWwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
T0*
_output_shapes
: 
я
Wwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Rank/SwitchSwitchwhile/Reshape_1while/cond/pred_id*
T0*"
_class
loc:@while/Reshape_1*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ

Qwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/sub/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 

Owhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/subSubPwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/RankQwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/sub/y*
T0*
_output_shapes
: 

Wwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range/startConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 

Wwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range/deltaConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 

Qwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/RangeRangeWwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range/startPwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/RankWwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range/delta*#
_output_shapes
:џџџџџџџџџ
Њ
Qwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/sub_1SubOwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/subQwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range*
T0*#
_output_shapes
:џџџџџџџџџ
Ж
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose	TransposeWwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Rank/SwitchQwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/sub_1*
T0*'
_output_shapes
:џџџџџџџџџ

Iwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstackUnpackKwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose*	
num*
T0*P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ

Ewhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/subSubKwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:3Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:1*#
_output_shapes
:џџџџџџџџџ*
T0

Gwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/sub_1SubKwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:2Iwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack*
T0*#
_output_shapes
:џџџџџџџџџ

Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 

Iwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truedivRealDivGwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/sub_1Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv/y*
T0*#
_output_shapes
:џџџџџџџџџ

Ewhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/addAddIwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstackIwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv*
T0*#
_output_shapes
:џџџџџџџџџ

Mwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv_1/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
valueB
 *   @

Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv_1RealDivEwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/subMwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv_1/y*
T0*#
_output_shapes
:џџџџџџџџџ

Gwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/add_1AddKwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:1Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ
{
/while/cond/batch_decode_2/Decode/transpose/RankRank!while/cond/batch_decode_2/unstack*
T0*
_output_shapes
: 
т
0while/cond/batch_decode_2/Decode/transpose/sub/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
_output_shapes
: *
value	B :*
dtype0
Й
.while/cond/batch_decode_2/Decode/transpose/subSub/while/cond/batch_decode_2/Decode/transpose/Rank0while/cond/batch_decode_2/Decode/transpose/sub/y*
T0*
_output_shapes
: 
ш
6while/cond/batch_decode_2/Decode/transpose/Range/startConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
ш
6while/cond/batch_decode_2/Decode/transpose/Range/deltaConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
_output_shapes
: *
value	B :*
dtype0
џ
0while/cond/batch_decode_2/Decode/transpose/RangeRange6while/cond/batch_decode_2/Decode/transpose/Range/start/while/cond/batch_decode_2/Decode/transpose/Rank6while/cond/batch_decode_2/Decode/transpose/Range/delta*#
_output_shapes
:џџџџџџџџџ
Ч
0while/cond/batch_decode_2/Decode/transpose/sub_1Sub.while/cond/batch_decode_2/Decode/transpose/sub0while/cond/batch_decode_2/Decode/transpose/Range*#
_output_shapes
:џџџџџџџџџ*
T0
О
*while/cond/batch_decode_2/Decode/transpose	Transpose!while/cond/batch_decode_2/unstack0while/cond/batch_decode_2/Decode/transpose/sub_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
(while/cond/batch_decode_2/Decode/unstackUnpack*while/cond/batch_decode_2/Decode/transpose*	
num*
T0*P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ

$while/cond/batch_decode_2/Decode/ExpExp*while/cond/batch_decode_2/Decode/unstack:3*
T0*#
_output_shapes
:џџџџџџџџџ
Ц
$while/cond/batch_decode_2/Decode/mulMul$while/cond/batch_decode_2/Decode/ExpEwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/sub*
T0*#
_output_shapes
:џџџџџџџџџ

&while/cond/batch_decode_2/Decode/Exp_1Exp*while/cond/batch_decode_2/Decode/unstack:2*
T0*#
_output_shapes
:џџџџџџџџџ
Ь
&while/cond/batch_decode_2/Decode/mul_1Mul&while/cond/batch_decode_2/Decode/Exp_1Gwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/sub_1*#
_output_shapes
:џџџџџџџџџ*
T0
Ю
&while/cond/batch_decode_2/Decode/mul_2Mul(while/cond/batch_decode_2/Decode/unstackGwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/sub_1*#
_output_shapes
:џџџџџџџџџ*
T0
Ш
$while/cond/batch_decode_2/Decode/addAdd&while/cond/batch_decode_2/Decode/mul_2Ewhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/add*#
_output_shapes
:џџџџџџџџџ*
T0
Ю
&while/cond/batch_decode_2/Decode/mul_3Mul*while/cond/batch_decode_2/Decode/unstack:1Ewhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/sub*
T0*#
_output_shapes
:џџџџџџџџџ
Ь
&while/cond/batch_decode_2/Decode/add_1Add&while/cond/batch_decode_2/Decode/mul_3Gwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/add_1*
T0*#
_output_shapes
:џџџџџџџџџ
п
*while/cond/batch_decode_2/Decode/truediv/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 
Е
(while/cond/batch_decode_2/Decode/truedivRealDiv&while/cond/batch_decode_2/Decode/mul_1*while/cond/batch_decode_2/Decode/truediv/y*#
_output_shapes
:џџџџџџџџџ*
T0
Љ
$while/cond/batch_decode_2/Decode/subSub$while/cond/batch_decode_2/Decode/add(while/cond/batch_decode_2/Decode/truediv*
T0*#
_output_shapes
:џџџџџџџџџ
с
,while/cond/batch_decode_2/Decode/truediv_1/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 
З
*while/cond/batch_decode_2/Decode/truediv_1RealDiv$while/cond/batch_decode_2/Decode/mul,while/cond/batch_decode_2/Decode/truediv_1/y*#
_output_shapes
:џџџџџџџџџ*
T0
Џ
&while/cond/batch_decode_2/Decode/sub_1Sub&while/cond/batch_decode_2/Decode/add_1*while/cond/batch_decode_2/Decode/truediv_1*
T0*#
_output_shapes
:џџџџџџџџџ
с
,while/cond/batch_decode_2/Decode/truediv_2/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 
Й
*while/cond/batch_decode_2/Decode/truediv_2RealDiv&while/cond/batch_decode_2/Decode/mul_1,while/cond/batch_decode_2/Decode/truediv_2/y*
T0*#
_output_shapes
:џџџџџџџџџ
­
&while/cond/batch_decode_2/Decode/add_2Add$while/cond/batch_decode_2/Decode/add*while/cond/batch_decode_2/Decode/truediv_2*#
_output_shapes
:џџџџџџџџџ*
T0
с
,while/cond/batch_decode_2/Decode/truediv_3/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB
 *   @*
dtype0*
_output_shapes
: 
З
*while/cond/batch_decode_2/Decode/truediv_3RealDiv$while/cond/batch_decode_2/Decode/mul,while/cond/batch_decode_2/Decode/truediv_3/y*#
_output_shapes
:џџџџџџџџџ*
T0
Џ
&while/cond/batch_decode_2/Decode/add_3Add&while/cond/batch_decode_2/Decode/add_1*while/cond/batch_decode_2/Decode/truediv_3*
T0*#
_output_shapes
:џџџџџџџџџ

&while/cond/batch_decode_2/Decode/stackPack$while/cond/batch_decode_2/Decode/sub&while/cond/batch_decode_2/Decode/sub_1&while/cond/batch_decode_2/Decode/add_2&while/cond/batch_decode_2/Decode/add_3*
T0*
N*'
_output_shapes
:џџџџџџџџџ

1while/cond/batch_decode_2/Decode/transpose_1/RankRank&while/cond/batch_decode_2/Decode/stack*
T0*
_output_shapes
: 
ф
2while/cond/batch_decode_2/Decode/transpose_1/sub/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
_output_shapes
: *
value	B :*
dtype0
П
0while/cond/batch_decode_2/Decode/transpose_1/subSub1while/cond/batch_decode_2/Decode/transpose_1/Rank2while/cond/batch_decode_2/Decode/transpose_1/sub/y*
T0*
_output_shapes
: 
ъ
8while/cond/batch_decode_2/Decode/transpose_1/Range/startConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
value	B : 
ъ
8while/cond/batch_decode_2/Decode/transpose_1/Range/deltaConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
value	B :

2while/cond/batch_decode_2/Decode/transpose_1/RangeRange8while/cond/batch_decode_2/Decode/transpose_1/Range/start1while/cond/batch_decode_2/Decode/transpose_1/Rank8while/cond/batch_decode_2/Decode/transpose_1/Range/delta*#
_output_shapes
:џџџџџџџџџ
Э
2while/cond/batch_decode_2/Decode/transpose_1/sub_1Sub0while/cond/batch_decode_2/Decode/transpose_1/sub2while/cond/batch_decode_2/Decode/transpose_1/Range*
T0*#
_output_shapes
:џџџџџџџџџ
Ч
,while/cond/batch_decode_2/Decode/transpose_1	Transpose&while/cond/batch_decode_2/Decode/stack2while/cond/batch_decode_2/Decode/transpose_1/sub_1*'
_output_shapes
:џџџџџџџџџ*
T0

while/cond/batch_decode_2/stackPack,while/cond/batch_decode_2/Decode/transpose_1*
T0*
N*+
_output_shapes
:џџџџџџџџџ
ё
7while/cond/batch_decode_2/nms_batch/strided_slice/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ѓ
9while/cond/batch_decode_2/nms_batch/strided_slice/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
ѓ
9while/cond/batch_decode_2/nms_batch/strided_slice/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
ш
1while/cond/batch_decode_2/nms_batch/strided_sliceStridedSlicewhile/cond/batch_decode_2/stack7while/cond/batch_decode_2/nms_batch/strided_slice/stack9while/cond/batch_decode_2/nms_batch/strided_slice/stack_19while/cond/batch_decode_2/nms_batch/strided_slice/stack_2*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask*
T0*
Index0
ѓ
9while/cond/batch_decode_2/nms_batch/strided_slice_1/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_2/nms_batch/strided_slice_1/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_2/nms_batch/strided_slice_1/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
№
3while/cond/batch_decode_2/nms_batch/strided_slice_1StridedSlice#while/cond/batch_decode_2/Reshape_19while/cond/batch_decode_2/nms_batch/strided_slice_1/stack;while/cond/batch_decode_2/nms_batch/strided_slice_1/stack_1;while/cond/batch_decode_2/nms_batch/strided_slice_1/stack_2*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask*
T0*
Index0
ъ
5while/cond/batch_decode_2/nms_batch/nms/iou_thresholdConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 
ь
7while/cond/batch_decode_2/nms_batch/nms/score_thresholdConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
_output_shapes
: *
valueB
 *
з#<*
dtype0
ж
/while/cond/batch_decode_2/nms_batch/nms/GreaterGreater3while/cond/batch_decode_2/nms_batch/strided_slice_17while/cond/batch_decode_2/nms_batch/nms/score_threshold*#
_output_shapes
:џџџџџџџџџ*
T0

:while/cond/batch_decode_2/nms_batch/nms/boolean_mask/ShapeShape1while/cond/batch_decode_2/nms_batch/strided_slice*
_output_shapes
:*
T0

Hwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Jwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

Jwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
Ђ
Bwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_sliceStridedSlice:while/cond/batch_decode_2/nms_batch/nms/boolean_mask/ShapeHwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stackJwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stack_1Jwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stack_2*
_output_shapes
:*
T0*
Index0

Kwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/Prod/reduction_indicesConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ѓ
9while/cond/batch_decode_2/nms_batch/nms/boolean_mask/ProdProdBwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_sliceKwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0

<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Shape_1Shape1while/cond/batch_decode_2/nms_batch/strided_slice*
T0*
_output_shapes
:

Jwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB:*
dtype0
М
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1StridedSlice<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Shape_1Jwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stackLwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stack_1Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stack_2*

begin_mask*
Index0*
T0*
_output_shapes
: 

<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Shape_2Shape1while/cond/batch_decode_2/nms_batch/strided_slice*
T0*
_output_shapes
:

Jwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:

Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
М
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2StridedSlice<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Shape_2Jwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stackLwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stack_1Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stack_2*
Index0*
T0*
end_mask*
_output_shapes
:
Е
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat/values_1Pack9while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Prod*
T0*
N*
_output_shapes
:
ђ
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat/axisConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 

;while/cond/batch_decode_2/nms_batch/nms/boolean_mask/concatConcatV2Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat/values_1Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2@while/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat/axis*
T0*
N*
_output_shapes
:
щ
<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/ReshapeReshape1while/cond/batch_decode_2/nms_batch/strided_slice;while/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat*
T0*'
_output_shapes
:џџџџџџџџџ

Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape_1/shapeConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
ю
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape_1Reshape/while/cond/batch_decode_2/nms_batch/nms/GreaterDwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape_1/shape*
T0
*#
_output_shapes
:џџџџџџџџџ
Ќ
:while/cond/batch_decode_2/nms_batch/nms/boolean_mask/WhereWhere>while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ш
<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/SqueezeSqueeze:while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Where*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims

є
Bwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/GatherV2/axisConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
о
=while/cond/batch_decode_2/nms_batch/nms/boolean_mask/GatherV2GatherV2<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/SqueezeBwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/GatherV2/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0

<while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/ShapeShape3while/cond/batch_decode_2/nms_batch/strided_slice_1*
T0*
_output_shapes
:

Jwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB: 

Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
Ќ
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_sliceStridedSlice<while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/ShapeJwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stackLwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stack_1Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:

Mwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Prod/reduction_indicesConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
љ
;while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/ProdProdDwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_sliceMwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Prod/reduction_indices*
T0*
_output_shapes
: 
Ё
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Shape_1Shape3while/cond/batch_decode_2/nms_batch/strided_slice_1*
_output_shapes
:*
T0

Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
Ц
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1StridedSlice>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Shape_1Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stackNwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2*
Index0*
T0*

begin_mask*
_output_shapes
: 
Ё
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Shape_2Shape3while/cond/batch_decode_2/nms_batch/strided_slice_1*
T0*
_output_shapes
:

Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:

Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:

Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
Ф
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2StridedSlice>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Shape_2Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stackNwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2*
end_mask*
_output_shapes
: *
T0*
Index0
Й
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat/values_1Pack;while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Prod*
T0*
N*
_output_shapes
:
є
Bwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat/axisConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
value	B : 

=while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concatConcatV2Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat/values_1Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2Bwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat/axis*
T0*
N*
_output_shapes
:
ы
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/ReshapeReshape3while/cond/batch_decode_2/nms_batch/strided_slice_1=while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat*
T0*#
_output_shapes
:џџџџџџџџџ

Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape_1/shapeConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ђ
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape_1Reshape/while/cond/batch_decode_2/nms_batch/nms/GreaterFwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape_1/shape*#
_output_shapes
:џџџџџџџџџ*
T0

А
<while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/WhereWhere@while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ь
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/SqueezeSqueeze<while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Where*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ
і
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/GatherV2/axisConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
т
?while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/GatherV2GatherV2>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/SqueezeDwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/GatherV2/axis*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	

-while/cond/batch_decode_2/nms_batch/nms/WhereWhere/while/cond/batch_decode_2/nms_batch/nms/Greater*'
_output_shapes
:џџџџџџџџџ
ј
5while/cond/batch_decode_2/nms_batch/nms/Reshape/shapeConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
Ю
/while/cond/batch_decode_2/nms_batch/nms/ReshapeReshape-while/cond/batch_decode_2/nms_batch/nms/Where5while/cond/batch_decode_2/nms_batch/nms/Reshape/shape*#
_output_shapes
:џџџџџџџџџ*
T0	
ў
Kwhile/cond/batch_decode_2/nms_batch/nms/NonMaxSuppressionV2/max_output_sizeConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value
B :Ш*
dtype0*
_output_shapes
: 
ћ
;while/cond/batch_decode_2/nms_batch/nms/NonMaxSuppressionV2NonMaxSuppressionV2=while/cond/batch_decode_2/nms_batch/nms/boolean_mask/GatherV2?while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/GatherV2Kwhile/cond/batch_decode_2/nms_batch/nms/NonMaxSuppressionV2/max_output_size5while/cond/batch_decode_2/nms_batch/nms/iou_threshold*#
_output_shapes
:џџџџџџџџџ
ч
5while/cond/batch_decode_2/nms_batch/nms/GatherV2/axisConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
В
0while/cond/batch_decode_2/nms_batch/nms/GatherV2GatherV2/while/cond/batch_decode_2/nms_batch/nms/Reshape;while/cond/batch_decode_2/nms_batch/nms/NonMaxSuppressionV25while/cond/batch_decode_2/nms_batch/nms/GatherV2/axis*
Tindices0*
Tparams0	*#
_output_shapes
:џџџџџџџџџ*
Taxis0
ѓ
9while/cond/batch_decode_2/nms_batch/strided_slice_2/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_2/nms_batch/strided_slice_2/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_2/nms_batch/strided_slice_2/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
№
3while/cond/batch_decode_2/nms_batch/strided_slice_2StridedSlicewhile/cond/batch_decode_2/stack9while/cond/batch_decode_2/nms_batch/strided_slice_2/stack;while/cond/batch_decode_2/nms_batch/strided_slice_2/stack_1;while/cond/batch_decode_2/nms_batch/strided_slice_2/stack_2*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask*
T0*
Index0
у
1while/cond/batch_decode_2/nms_batch/GatherV2/axisConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
Ї
,while/cond/batch_decode_2/nms_batch/GatherV2GatherV23while/cond/batch_decode_2/nms_batch/strided_slice_20while/cond/batch_decode_2/nms_batch/nms/GatherV21while/cond/batch_decode_2/nms_batch/GatherV2/axis*
Taxis0*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџ
ѓ
9while/cond/batch_decode_2/nms_batch/strided_slice_3/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
ѕ
;while/cond/batch_decode_2/nms_batch/strided_slice_3/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB:*
dtype0
ѕ
;while/cond/batch_decode_2/nms_batch/strided_slice_3/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
№
3while/cond/batch_decode_2/nms_batch/strided_slice_3StridedSlice#while/cond/batch_decode_2/Reshape_19while/cond/batch_decode_2/nms_batch/strided_slice_3/stack;while/cond/batch_decode_2/nms_batch/strided_slice_3/stack_1;while/cond/batch_decode_2/nms_batch/strided_slice_3/stack_2*
shrink_axis_mask*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ
х
3while/cond/batch_decode_2/nms_batch/GatherV2_1/axisConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
Ї
.while/cond/batch_decode_2/nms_batch/GatherV2_1GatherV23while/cond/batch_decode_2/nms_batch/strided_slice_30while/cond/batch_decode_2/nms_batch/nms/GatherV23while/cond/batch_decode_2/nms_batch/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ

3while/cond/batch_decode_2/nms_batch/batch_pad/ShapeShape,while/cond/batch_decode_2/nms_batch/GatherV2*
T0*
_output_shapes
:
ћ
Awhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
§
Cwhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
§
Cwhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:

;while/cond/batch_decode_2/nms_batch/batch_pad/strided_sliceStridedSlice3while/cond/batch_decode_2/nms_batch/batch_pad/ShapeAwhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stackCwhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stack_1Cwhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
І
3while/cond/batch_decode_2/nms_batch/batch_pad/stackPack;while/cond/batch_decode_2/nms_batch/batch_pad/strided_slice*
T0*
N*
_output_shapes
:
э
3while/cond/batch_decode_2/nms_batch/batch_pad/ConstConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
У
1while/cond/batch_decode_2/nms_batch/batch_pad/MaxMax3while/cond/batch_decode_2/nms_batch/batch_pad/stack3while/cond/batch_decode_2/nms_batch/batch_pad/Const*
T0*
_output_shapes
: 

<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/ShapeShape,while/cond/batch_decode_2/nms_batch/GatherV2*
T0*
_output_shapes
:
Д
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/unstackUnpack<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Shape*	
num*
T0*
_output_shapes
: : 
ю
<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
г
:while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/subSub1while/cond/batch_decode_2/nms_batch/batch_pad/Max<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub/y*
T0*
_output_shapes
: 
р
<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub_1Sub:while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/unstack*
_output_shapes
: *
T0
ђ
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Maximum/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
value	B : 
ъ
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/MaximumMaximum<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub_1@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Maximum/y*
T0*
_output_shapes
: 
§
<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
:*
valueB"        
ђ
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_1/1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
_output_shapes
: *
value	B : *
dtype0
і
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_1Pack>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Maximum@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_1/1*
T0*
N*
_output_shapes
:

>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_2Pack<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_1*
T0*

axis*
N*
_output_shapes

:
ъ
:while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/PadPad,while/cond/batch_decode_2/nms_batch/GatherV2>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_2*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
ђ
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_3/1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
щ
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_3Pack1while/cond/batch_decode_2/nms_batch/batch_pad/Max@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_3/1*
N*
_output_shapes
:*
T0
ї
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/ReshapeReshape:while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Pad>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_3*
T0*'
_output_shapes
:џџџџџџџџџ
М
5while/cond/batch_decode_2/nms_batch/batch_pad/stack_1Pack>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Reshape*
T0*
N*+
_output_shapes
:џџџџџџџџџ

5while/cond/batch_decode_2/nms_batch/batch_pad_1/ShapeShape.while/cond/batch_decode_2/nms_batch/GatherV2_1*
T0*
_output_shapes
:
§
Cwhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
џ
Ewhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stack_1Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
џ
Ewhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stack_2Const5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB:*
dtype0

=while/cond/batch_decode_2/nms_batch/batch_pad_1/strided_sliceStridedSlice5while/cond/batch_decode_2/nms_batch/batch_pad_1/ShapeCwhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stackEwhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stack_1Ewhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
Њ
5while/cond/batch_decode_2/nms_batch/batch_pad_1/stackPack=while/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice*
T0*
N*
_output_shapes
:
я
5while/cond/batch_decode_2/nms_batch/batch_pad_1/ConstConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
_output_shapes
:*
valueB: *
dtype0
Щ
3while/cond/batch_decode_2/nms_batch/batch_pad_1/MaxMax5while/cond/batch_decode_2/nms_batch/batch_pad_1/stack5while/cond/batch_decode_2/nms_batch/batch_pad_1/Const*
_output_shapes
: *
T0

>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/ShapeShape.while/cond/batch_decode_2/nms_batch/GatherV2_1*
T0*
_output_shapes
:
Ж
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/unstackUnpack>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Shape*	
num*
T0*
_output_shapes
: 
№
>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
dtype0*
_output_shapes
: *
value	B : 
й
<while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/subSub3while/cond/batch_decode_2/nms_batch/batch_pad_1/Max>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub/y*
T0*
_output_shapes
: 
ц
>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub_1Sub<while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/unstack*
T0*
_output_shapes
: 
є
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Maximum/yConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
value	B : *
dtype0*
_output_shapes
: 
№
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/MaximumMaximum>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub_1Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Maximum/y*
T0*
_output_shapes
: 
ј
>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stackConst5^while/cond/batch_decode_2/assert_equal/Assert/Assert7^while/cond/batch_decode_2/assert_equal_1/Assert/Assert*
valueB: *
dtype0*
_output_shapes
:
И
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_1Pack@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Maximum*
_output_shapes
:*
T0*
N

@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_2Pack>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_1*
N*
_output_shapes

:*
T0*

axis
у
<while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/PadPad.while/cond/batch_decode_2/nms_batch/GatherV2_1@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_2*
T0*#
_output_shapes
:џџџџџџџџџ
Ћ
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_3Pack3while/cond/batch_decode_2/nms_batch/batch_pad_1/Max*
T0*
N*
_output_shapes
:
љ
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/ReshapeReshape<while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Pad@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_3*#
_output_shapes
:џџџџџџџџџ*
T0
М
7while/cond/batch_decode_2/nms_batch/batch_pad_1/stack_1Pack@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Reshape*
T0*
N*'
_output_shapes
:џџџџџџџџџ
І
while/cond/MergeMerge5while/cond/batch_decode_2/nms_batch/batch_pad/stack_1while/cond/concat_1*
N*-
_output_shapes
:џџџџџџџџџ: *
T0
І
while/cond/Merge_1Merge7while/cond/batch_decode_2/nms_batch/batch_pad_1/stack_1while/cond/concat_2*
N*)
_output_shapes
:џџџџџџџџџ: *
T0
x
%while/img_shape_3/assert_rank_in/rankConst^while/Identity*
dtype0*
_output_shapes
: *
value	B :
z
'while/img_shape_3/assert_rank_in/rank_1Const^while/Identity*
_output_shapes
: *
value	B :*
dtype0
m
&while/img_shape_3/assert_rank_in/ShapeShapewhile/TensorArrayReadV3*
T0*
_output_shapes
:
h
Owhile/img_shape_3/assert_rank_in/assert_type/statically_determined_correct_typeNoOp^while/Identity
j
Qwhile/img_shape_3/assert_rank_in/assert_type_1/statically_determined_correct_typeNoOp^while/Identity
Y
@while/img_shape_3/assert_rank_in/static_checks_determined_all_okNoOp^while/Identity

while/img_shape_3/RankConstA^while/img_shape_3/assert_rank_in/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 

while/img_shape_3/Equal/yConstA^while/img_shape_3/assert_rank_in/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 
t
while/img_shape_3/EqualEqualwhile/img_shape_3/Rankwhile/img_shape_3/Equal/y*
T0*
_output_shapes
: 
|
while/img_shape_3/cond/SwitchSwitchwhile/img_shape_3/Equalwhile/img_shape_3/Equal*
T0
*
_output_shapes
: : 
m
while/img_shape_3/cond/switch_tIdentitywhile/img_shape_3/cond/Switch:1*
T0
*
_output_shapes
: 
k
while/img_shape_3/cond/switch_fIdentitywhile/img_shape_3/cond/Switch*
_output_shapes
: *
T0

d
while/img_shape_3/cond/pred_idIdentitywhile/img_shape_3/Equal*
T0
*
_output_shapes
: 
Д
while/img_shape_3/cond/ShapeShape%while/img_shape_3/cond/Shape/Switch:1A^while/img_shape_3/assert_rank_in/static_checks_determined_all_ok*
_output_shapes
:*
T0
љ
#while/img_shape_3/cond/Shape/SwitchSwitchwhile/TensorArrayReadV3while/img_shape_3/cond/pred_id*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ*
T0**
_class 
loc:@while/TensorArrayReadV3
й
*while/img_shape_3/cond/strided_slice/stackConstA^while/img_shape_3/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_3/cond/switch_t*
_output_shapes
:*
valueB: *
dtype0
л
,while/img_shape_3/cond/strided_slice/stack_1ConstA^while/img_shape_3/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_3/cond/switch_t*
dtype0*
_output_shapes
:*
valueB:
л
,while/img_shape_3/cond/strided_slice/stack_2ConstA^while/img_shape_3/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_3/cond/switch_t*
valueB:*
dtype0*
_output_shapes
:

$while/img_shape_3/cond/strided_sliceStridedSlicewhile/img_shape_3/cond/Shape*while/img_shape_3/cond/strided_slice/stack,while/img_shape_3/cond/strided_slice/stack_1,while/img_shape_3/cond/strided_slice/stack_2*

begin_mask*
Index0*
T0*
_output_shapes
:
Ж
while/img_shape_3/cond/Shape_1Shape%while/img_shape_3/cond/Shape_1/SwitchA^while/img_shape_3/assert_rank_in/static_checks_determined_all_ok*
T0*
_output_shapes
:
ћ
%while/img_shape_3/cond/Shape_1/SwitchSwitchwhile/TensorArrayReadV3while/img_shape_3/cond/pred_id*
T0**
_class 
loc:@while/TensorArrayReadV3*\
_output_shapesJ
H:"џџџџџџџџџџџџџџџџџџ:"џџџџџџџџџџџџџџџџџџ
л
,while/img_shape_3/cond/strided_slice_1/stackConstA^while/img_shape_3/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_3/cond/switch_f*
dtype0*
_output_shapes
:*
valueB:
н
.while/img_shape_3/cond/strided_slice_1/stack_1ConstA^while/img_shape_3/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_3/cond/switch_f*
dtype0*
_output_shapes
:*
valueB:
н
.while/img_shape_3/cond/strided_slice_1/stack_2ConstA^while/img_shape_3/assert_rank_in/static_checks_determined_all_ok ^while/img_shape_3/cond/switch_f*
valueB:*
dtype0*
_output_shapes
:

&while/img_shape_3/cond/strided_slice_1StridedSlicewhile/img_shape_3/cond/Shape_1,while/img_shape_3/cond/strided_slice_1/stack.while/img_shape_3/cond/strided_slice_1/stack_1.while/img_shape_3/cond/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
:
Ѓ
while/img_shape_3/cond/MergeMerge&while/img_shape_3/cond/strided_slice_1$while/img_shape_3/cond/strided_slice*
_output_shapes

:: *
T0*
N
f
while/Cast_4Castwhile/img_shape_3/cond/Merge*

SrcT0*
_output_shapes
:*

DstT0
}
while/strided_slice_8/stackConst^while/Identity*
_output_shapes
:*
valueB"        *
dtype0

while/strided_slice_8/stack_1Const^while/Identity*
valueB"       *
dtype0*
_output_shapes
:

while/strided_slice_8/stack_2Const^while/Identity*
dtype0*
_output_shapes
:*
valueB"      
ў
while/strided_slice_8StridedSlicewhile/cond/Mergewhile/strided_slice_8/stackwhile/strided_slice_8/stack_1while/strided_slice_8/stack_2*
ellipsis_mask*
T0*
Index0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
v
while/strided_slice_9/stackConst^while/Identity*
valueB: *
dtype0*
_output_shapes
:
x
while/strided_slice_9/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
x
while/strided_slice_9/stack_2Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
д
while/strided_slice_9StridedSlicewhile/Cast_4while/strided_slice_9/stackwhile/strided_slice_9/stack_1while/strided_slice_9/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
z
while/truediv_5RealDivwhile/strided_slice_8while/strided_slice_9*'
_output_shapes
:џџџџџџџџџ*
T0
~
while/strided_slice_10/stackConst^while/Identity*
valueB"       *
dtype0*
_output_shapes
:

while/strided_slice_10/stack_1Const^while/Identity*
valueB"       *
dtype0*
_output_shapes
:

while/strided_slice_10/stack_2Const^while/Identity*
valueB"      *
dtype0*
_output_shapes
:

while/strided_slice_10StridedSlicewhile/cond/Mergewhile/strided_slice_10/stackwhile/strided_slice_10/stack_1while/strided_slice_10/stack_2*
shrink_axis_mask*
T0*
Index0*
ellipsis_mask*'
_output_shapes
:џџџџџџџџџ
w
while/strided_slice_11/stackConst^while/Identity*
valueB:*
dtype0*
_output_shapes
:
y
while/strided_slice_11/stack_1Const^while/Identity*
_output_shapes
:*
valueB:*
dtype0
y
while/strided_slice_11/stack_2Const^while/Identity*
_output_shapes
:*
valueB:*
dtype0
и
while/strided_slice_11StridedSlicewhile/Cast_4while/strided_slice_11/stackwhile/strided_slice_11/stack_1while/strided_slice_11/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
|
while/truediv_6RealDivwhile/strided_slice_10while/strided_slice_11*'
_output_shapes
:џџџџџџџџџ*
T0
~
while/strided_slice_12/stackConst^while/Identity*
valueB"       *
dtype0*
_output_shapes
:

while/strided_slice_12/stack_1Const^while/Identity*
valueB"       *
dtype0*
_output_shapes
:

while/strided_slice_12/stack_2Const^while/Identity*
dtype0*
_output_shapes
:*
valueB"      

while/strided_slice_12StridedSlicewhile/cond/Mergewhile/strided_slice_12/stackwhile/strided_slice_12/stack_1while/strided_slice_12/stack_2*
shrink_axis_mask*
ellipsis_mask*
T0*
Index0*'
_output_shapes
:џџџџџџџџџ
w
while/strided_slice_13/stackConst^while/Identity*
valueB: *
dtype0*
_output_shapes
:
y
while/strided_slice_13/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
y
while/strided_slice_13/stack_2Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
и
while/strided_slice_13StridedSlicewhile/Cast_4while/strided_slice_13/stackwhile/strided_slice_13/stack_1while/strided_slice_13/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
|
while/truediv_7RealDivwhile/strided_slice_12while/strided_slice_13*
T0*'
_output_shapes
:џџџџџџџџџ
~
while/strided_slice_14/stackConst^while/Identity*
valueB"       *
dtype0*
_output_shapes
:

while/strided_slice_14/stack_1Const^while/Identity*
valueB"       *
dtype0*
_output_shapes
:

while/strided_slice_14/stack_2Const^while/Identity*
_output_shapes
:*
valueB"      *
dtype0

while/strided_slice_14StridedSlicewhile/cond/Mergewhile/strided_slice_14/stackwhile/strided_slice_14/stack_1while/strided_slice_14/stack_2*
ellipsis_mask*
T0*
Index0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
w
while/strided_slice_15/stackConst^while/Identity*
valueB:*
dtype0*
_output_shapes
:
y
while/strided_slice_15/stack_1Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
y
while/strided_slice_15/stack_2Const^while/Identity*
valueB:*
dtype0*
_output_shapes
:
и
while/strided_slice_15StridedSlicewhile/Cast_4while/strided_slice_15/stackwhile/strided_slice_15/stack_1while/strided_slice_15/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
|
while/truediv_8RealDivwhile/strided_slice_14while/strided_slice_15*'
_output_shapes
:џџџџџџџџџ*
T0
­
while/stack_4Packwhile/truediv_5while/truediv_6while/truediv_7while/truediv_8*
T0*
axisџџџџџџџџџ*
N*+
_output_shapes
:џџџџџџџџџ
d
while/concat/axisConst^while/Identity*
dtype0*
_output_shapes
: *
value	B :

while/concatConcatV2while/Identity_1while/stack_4while/concat/axis*
N*+
_output_shapes
:џџџџџџџџџ*
T0
f
while/concat_1/axisConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 

while/concat_1ConcatV2while/Identity_2while/cond/Merge_1while/concat_1/axis*
T0*
N*'
_output_shapes
:џџџџџџџџџ
`
while/add_5/yConst^while/Identity*
value	B :*
dtype0*
_output_shapes
: 
R
while/add_5Addwhile/Identitywhile/add_5/y*
T0*
_output_shapes
: 
R
while/NextIterationNextIterationwhile/add_5*
T0*
_output_shapes
: 
j
while/NextIteration_1NextIterationwhile/concat*
T0*+
_output_shapes
:џџџџџџџџџ
h
while/NextIteration_2NextIterationwhile/concat_1*
T0*'
_output_shapes
:џџџџџџџџџ
A

while/ExitExitwhile/Switch*
_output_shapes
: *
T0
Z
while/Exit_1Exitwhile/Switch_1*
T0*+
_output_shapes
:џџџџџџџџџ
V
while/Exit_2Exitwhile/Switch_2*
T0*'
_output_shapes
:џџџџџџџџџ
g
nms_batch/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
i
nms_batch/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
nms_batch/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
э
nms_batch/strided_sliceStridedSlicewhile/Exit_1nms_batch/strided_slice/stacknms_batch/strided_slice/stack_1nms_batch/strided_slice/stack_2*
T0*
Index0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
i
nms_batch/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: 
k
!nms_batch/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!nms_batch/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ё
nms_batch/strided_slice_1StridedSlicewhile/Exit_2nms_batch/strided_slice_1/stack!nms_batch/strided_slice_1/stack_1!nms_batch/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*#
_output_shapes
:џџџџџџџџџ
`
nms_batch/nms/iou_thresholdConst*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 
b
nms_batch/nms/score_thresholdConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 

nms_batch/nms/GreaterGreaternms_batch/strided_slice_1nms_batch/nms/score_threshold*
T0*#
_output_shapes
:џџџџџџџџџ
g
 nms_batch/nms/boolean_mask/ShapeShapenms_batch/strided_slice*
_output_shapes
:*
T0
x
.nms_batch/nms/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0nms_batch/nms/boolean_mask/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
z
0nms_batch/nms/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
(nms_batch/nms/boolean_mask/strided_sliceStridedSlice nms_batch/nms/boolean_mask/Shape.nms_batch/nms/boolean_mask/strided_slice/stack0nms_batch/nms/boolean_mask/strided_slice/stack_10nms_batch/nms/boolean_mask/strided_slice/stack_2*
T0*
Index0*
_output_shapes
:
{
1nms_batch/nms/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
Ѕ
nms_batch/nms/boolean_mask/ProdProd(nms_batch/nms/boolean_mask/strided_slice1nms_batch/nms/boolean_mask/Prod/reduction_indices*
T0*
_output_shapes
: 
i
"nms_batch/nms/boolean_mask/Shape_1Shapenms_batch/strided_slice*
T0*
_output_shapes
:
z
0nms_batch/nms/boolean_mask/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2nms_batch/nms/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
|
2nms_batch/nms/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
К
*nms_batch/nms/boolean_mask/strided_slice_1StridedSlice"nms_batch/nms/boolean_mask/Shape_10nms_batch/nms/boolean_mask/strided_slice_1/stack2nms_batch/nms/boolean_mask/strided_slice_1/stack_12nms_batch/nms/boolean_mask/strided_slice_1/stack_2*

begin_mask*
Index0*
T0*
_output_shapes
: 
i
"nms_batch/nms/boolean_mask/Shape_2Shapenms_batch/strided_slice*
T0*
_output_shapes
:
z
0nms_batch/nms/boolean_mask/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2nms_batch/nms/boolean_mask/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
|
2nms_batch/nms/boolean_mask/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
К
*nms_batch/nms/boolean_mask/strided_slice_2StridedSlice"nms_batch/nms/boolean_mask/Shape_20nms_batch/nms/boolean_mask/strided_slice_2/stack2nms_batch/nms/boolean_mask/strided_slice_2/stack_12nms_batch/nms/boolean_mask/strided_slice_2/stack_2*
Index0*
T0*
end_mask*
_output_shapes
:

*nms_batch/nms/boolean_mask/concat/values_1Packnms_batch/nms/boolean_mask/Prod*
T0*
N*
_output_shapes
:
h
&nms_batch/nms/boolean_mask/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

!nms_batch/nms/boolean_mask/concatConcatV2*nms_batch/nms/boolean_mask/strided_slice_1*nms_batch/nms/boolean_mask/concat/values_1*nms_batch/nms/boolean_mask/strided_slice_2&nms_batch/nms/boolean_mask/concat/axis*
N*
_output_shapes
:*
T0

"nms_batch/nms/boolean_mask/ReshapeReshapenms_batch/strided_slice!nms_batch/nms/boolean_mask/concat*'
_output_shapes
:џџџџџџџџџ*
T0
}
*nms_batch/nms/boolean_mask/Reshape_1/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
 
$nms_batch/nms/boolean_mask/Reshape_1Reshapenms_batch/nms/Greater*nms_batch/nms/boolean_mask/Reshape_1/shape*
T0
*#
_output_shapes
:џџџџџџџџџ
x
 nms_batch/nms/boolean_mask/WhereWhere$nms_batch/nms/boolean_mask/Reshape_1*'
_output_shapes
:џџџџџџџџџ

"nms_batch/nms/boolean_mask/SqueezeSqueeze nms_batch/nms/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ
j
(nms_batch/nms/boolean_mask/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
і
#nms_batch/nms/boolean_mask/GatherV2GatherV2"nms_batch/nms/boolean_mask/Reshape"nms_batch/nms/boolean_mask/Squeeze(nms_batch/nms/boolean_mask/GatherV2/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0
k
"nms_batch/nms/boolean_mask_1/ShapeShapenms_batch/strided_slice_1*
_output_shapes
:*
T0
z
0nms_batch/nms/boolean_mask_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2nms_batch/nms/boolean_mask_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2nms_batch/nms/boolean_mask_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Њ
*nms_batch/nms/boolean_mask_1/strided_sliceStridedSlice"nms_batch/nms/boolean_mask_1/Shape0nms_batch/nms/boolean_mask_1/strided_slice/stack2nms_batch/nms/boolean_mask_1/strided_slice/stack_12nms_batch/nms/boolean_mask_1/strided_slice/stack_2*
_output_shapes
:*
T0*
Index0
}
3nms_batch/nms/boolean_mask_1/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
Ћ
!nms_batch/nms/boolean_mask_1/ProdProd*nms_batch/nms/boolean_mask_1/strided_slice3nms_batch/nms/boolean_mask_1/Prod/reduction_indices*
T0*
_output_shapes
: 
m
$nms_batch/nms/boolean_mask_1/Shape_1Shapenms_batch/strided_slice_1*
T0*
_output_shapes
:
|
2nms_batch/nms/boolean_mask_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
4nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
~
4nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ф
,nms_batch/nms/boolean_mask_1/strided_slice_1StridedSlice$nms_batch/nms/boolean_mask_1/Shape_12nms_batch/nms/boolean_mask_1/strided_slice_1/stack4nms_batch/nms/boolean_mask_1/strided_slice_1/stack_14nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2*
T0*
Index0*

begin_mask*
_output_shapes
: 
m
$nms_batch/nms/boolean_mask_1/Shape_2Shapenms_batch/strided_slice_1*
_output_shapes
:*
T0
|
2nms_batch/nms/boolean_mask_1/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
~
4nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Т
,nms_batch/nms/boolean_mask_1/strided_slice_2StridedSlice$nms_batch/nms/boolean_mask_1/Shape_22nms_batch/nms/boolean_mask_1/strided_slice_2/stack4nms_batch/nms/boolean_mask_1/strided_slice_2/stack_14nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2*
Index0*
T0*
end_mask*
_output_shapes
: 

,nms_batch/nms/boolean_mask_1/concat/values_1Pack!nms_batch/nms/boolean_mask_1/Prod*
N*
_output_shapes
:*
T0
j
(nms_batch/nms/boolean_mask_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 

#nms_batch/nms/boolean_mask_1/concatConcatV2,nms_batch/nms/boolean_mask_1/strided_slice_1,nms_batch/nms/boolean_mask_1/concat/values_1,nms_batch/nms/boolean_mask_1/strided_slice_2(nms_batch/nms/boolean_mask_1/concat/axis*
T0*
N*
_output_shapes
:

$nms_batch/nms/boolean_mask_1/ReshapeReshapenms_batch/strided_slice_1#nms_batch/nms/boolean_mask_1/concat*
T0*#
_output_shapes
:џџџџџџџџџ

,nms_batch/nms/boolean_mask_1/Reshape_1/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
Є
&nms_batch/nms/boolean_mask_1/Reshape_1Reshapenms_batch/nms/Greater,nms_batch/nms/boolean_mask_1/Reshape_1/shape*
T0
*#
_output_shapes
:џџџџџџџџџ
|
"nms_batch/nms/boolean_mask_1/WhereWhere&nms_batch/nms/boolean_mask_1/Reshape_1*'
_output_shapes
:џџџџџџџџџ

$nms_batch/nms/boolean_mask_1/SqueezeSqueeze"nms_batch/nms/boolean_mask_1/Where*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ
l
*nms_batch/nms/boolean_mask_1/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
њ
%nms_batch/nms/boolean_mask_1/GatherV2GatherV2$nms_batch/nms/boolean_mask_1/Reshape$nms_batch/nms/boolean_mask_1/Squeeze*nms_batch/nms/boolean_mask_1/GatherV2/axis*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ
\
nms_batch/nms/WhereWherenms_batch/nms/Greater*'
_output_shapes
:џџџџџџџџџ
n
nms_batch/nms/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:

nms_batch/nms/ReshapeReshapenms_batch/nms/Wherenms_batch/nms/Reshape/shape*#
_output_shapes
:џџџџџџџџџ*
T0	
t
1nms_batch/nms/NonMaxSuppressionV2/max_output_sizeConst*
value
B :Ш*
dtype0*
_output_shapes
: 
љ
!nms_batch/nms/NonMaxSuppressionV2NonMaxSuppressionV2#nms_batch/nms/boolean_mask/GatherV2%nms_batch/nms/boolean_mask_1/GatherV21nms_batch/nms/NonMaxSuppressionV2/max_output_sizenms_batch/nms/iou_threshold*#
_output_shapes
:џџџџџџџџџ
]
nms_batch/nms/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ъ
nms_batch/nms/GatherV2GatherV2nms_batch/nms/Reshape!nms_batch/nms/NonMaxSuppressionV2nms_batch/nms/GatherV2/axis*
Tparams0	*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0
i
nms_batch/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
k
!nms_batch/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!nms_batch/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ѕ
nms_batch/strided_slice_2StridedSlicewhile/Exit_1nms_batch/strided_slice_2/stack!nms_batch/strided_slice_2/stack_1!nms_batch/strided_slice_2/stack_2*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
Y
nms_batch/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0
П
nms_batch/GatherV2GatherV2nms_batch/strided_slice_2nms_batch/nms/GatherV2nms_batch/GatherV2/axis*
Tindices0	*
Tparams0*'
_output_shapes
:џџџџџџџџџ*
Taxis0
i
nms_batch/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
k
!nms_batch/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!nms_batch/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ё
nms_batch/strided_slice_3StridedSlicewhile/Exit_2nms_batch/strided_slice_3/stack!nms_batch/strided_slice_3/stack_1!nms_batch/strided_slice_3/stack_2*
T0*
Index0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
[
nms_batch/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
П
nms_batch/GatherV2_1GatherV2nms_batch/strided_slice_3nms_batch/nms/GatherV2nms_batch/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ
[
nms_batch/batch_pad/ShapeShapenms_batch/GatherV2*
T0*
_output_shapes
:
q
'nms_batch/batch_pad/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)nms_batch/batch_pad/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)nms_batch/batch_pad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

!nms_batch/batch_pad/strided_sliceStridedSlicenms_batch/batch_pad/Shape'nms_batch/batch_pad/strided_slice/stack)nms_batch/batch_pad/strided_slice/stack_1)nms_batch/batch_pad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
r
nms_batch/batch_pad/stackPack!nms_batch/batch_pad/strided_slice*
T0*
N*
_output_shapes
:
c
nms_batch/batch_pad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
u
nms_batch/batch_pad/MaxMaxnms_batch/batch_pad/stacknms_batch/batch_pad/Const*
T0*
_output_shapes
: 
d
"nms_batch/batch_pad/pad_axis/ShapeShapenms_batch/GatherV2*
T0*
_output_shapes
:

$nms_batch/batch_pad/pad_axis/unstackUnpack"nms_batch/batch_pad/pad_axis/Shape*	
num*
T0*
_output_shapes
: : 
d
"nms_batch/batch_pad/pad_axis/sub/yConst*
value	B : *
dtype0*
_output_shapes
: 

 nms_batch/batch_pad/pad_axis/subSubnms_batch/batch_pad/Max"nms_batch/batch_pad/pad_axis/sub/y*
T0*
_output_shapes
: 

"nms_batch/batch_pad/pad_axis/sub_1Sub nms_batch/batch_pad/pad_axis/sub$nms_batch/batch_pad/pad_axis/unstack*
_output_shapes
: *
T0
h
&nms_batch/batch_pad/pad_axis/Maximum/yConst*
value	B : *
dtype0*
_output_shapes
: 

$nms_batch/batch_pad/pad_axis/MaximumMaximum"nms_batch/batch_pad/pad_axis/sub_1&nms_batch/batch_pad/pad_axis/Maximum/y*
T0*
_output_shapes
: 
s
"nms_batch/batch_pad/pad_axis/stackConst*
valueB"        *
dtype0*
_output_shapes
:
h
&nms_batch/batch_pad/pad_axis/stack_1/1Const*
value	B : *
dtype0*
_output_shapes
: 
Ј
$nms_batch/batch_pad/pad_axis/stack_1Pack$nms_batch/batch_pad/pad_axis/Maximum&nms_batch/batch_pad/pad_axis/stack_1/1*
T0*
N*
_output_shapes
:
Д
$nms_batch/batch_pad/pad_axis/stack_2Pack"nms_batch/batch_pad/pad_axis/stack$nms_batch/batch_pad/pad_axis/stack_1*
T0*

axis*
N*
_output_shapes

:

 nms_batch/batch_pad/pad_axis/PadPadnms_batch/GatherV2$nms_batch/batch_pad/pad_axis/stack_2*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
h
&nms_batch/batch_pad/pad_axis/stack_3/1Const*
dtype0*
_output_shapes
: *
value	B :

$nms_batch/batch_pad/pad_axis/stack_3Packnms_batch/batch_pad/Max&nms_batch/batch_pad/pad_axis/stack_3/1*
T0*
N*
_output_shapes
:
Љ
$nms_batch/batch_pad/pad_axis/ReshapeReshape nms_batch/batch_pad/pad_axis/Pad$nms_batch/batch_pad/pad_axis/stack_3*'
_output_shapes
:џџџџџџџџџ*
T0

nms_batch/batch_pad/stack_1Pack$nms_batch/batch_pad/pad_axis/Reshape*
T0*
N*+
_output_shapes
:џџџџџџџџџ
_
nms_batch/batch_pad_1/ShapeShapenms_batch/GatherV2_1*
_output_shapes
:*
T0
s
)nms_batch/batch_pad_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+nms_batch/batch_pad_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+nms_batch/batch_pad_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

#nms_batch/batch_pad_1/strided_sliceStridedSlicenms_batch/batch_pad_1/Shape)nms_batch/batch_pad_1/strided_slice/stack+nms_batch/batch_pad_1/strided_slice/stack_1+nms_batch/batch_pad_1/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
v
nms_batch/batch_pad_1/stackPack#nms_batch/batch_pad_1/strided_slice*
T0*
N*
_output_shapes
:
e
nms_batch/batch_pad_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
nms_batch/batch_pad_1/MaxMaxnms_batch/batch_pad_1/stacknms_batch/batch_pad_1/Const*
T0*
_output_shapes
: 
h
$nms_batch/batch_pad_1/pad_axis/ShapeShapenms_batch/GatherV2_1*
T0*
_output_shapes
:

&nms_batch/batch_pad_1/pad_axis/unstackUnpack$nms_batch/batch_pad_1/pad_axis/Shape*
_output_shapes
: *	
num*
T0
f
$nms_batch/batch_pad_1/pad_axis/sub/yConst*
value	B : *
dtype0*
_output_shapes
: 

"nms_batch/batch_pad_1/pad_axis/subSubnms_batch/batch_pad_1/Max$nms_batch/batch_pad_1/pad_axis/sub/y*
_output_shapes
: *
T0

$nms_batch/batch_pad_1/pad_axis/sub_1Sub"nms_batch/batch_pad_1/pad_axis/sub&nms_batch/batch_pad_1/pad_axis/unstack*
T0*
_output_shapes
: 
j
(nms_batch/batch_pad_1/pad_axis/Maximum/yConst*
value	B : *
dtype0*
_output_shapes
: 
Ђ
&nms_batch/batch_pad_1/pad_axis/MaximumMaximum$nms_batch/batch_pad_1/pad_axis/sub_1(nms_batch/batch_pad_1/pad_axis/Maximum/y*
T0*
_output_shapes
: 
n
$nms_batch/batch_pad_1/pad_axis/stackConst*
valueB: *
dtype0*
_output_shapes
:

&nms_batch/batch_pad_1/pad_axis/stack_1Pack&nms_batch/batch_pad_1/pad_axis/Maximum*
_output_shapes
:*
T0*
N
К
&nms_batch/batch_pad_1/pad_axis/stack_2Pack$nms_batch/batch_pad_1/pad_axis/stack&nms_batch/batch_pad_1/pad_axis/stack_1*
T0*

axis*
N*
_output_shapes

:

"nms_batch/batch_pad_1/pad_axis/PadPadnms_batch/GatherV2_1&nms_batch/batch_pad_1/pad_axis/stack_2*#
_output_shapes
:џџџџџџџџџ*
T0
w
&nms_batch/batch_pad_1/pad_axis/stack_3Packnms_batch/batch_pad_1/Max*
_output_shapes
:*
T0*
N
Ћ
&nms_batch/batch_pad_1/pad_axis/ReshapeReshape"nms_batch/batch_pad_1/pad_axis/Pad&nms_batch/batch_pad_1/pad_axis/stack_3*
T0*#
_output_shapes
:џџџџџџџџџ

nms_batch/batch_pad_1/stack_1Pack&nms_batch/batch_pad_1/pad_axis/Reshape*
T0*
N*'
_output_shapes
:џџџџџџџџџ
i
nms_batch_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
k
!nms_batch_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
k
!nms_batch_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

nms_batch_1/strided_sliceStridedSlicenms_batch/batch_pad/stack_1nms_batch_1/strided_slice/stack!nms_batch_1/strided_slice/stack_1!nms_batch_1/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
k
!nms_batch_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
m
#nms_batch_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#nms_batch_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

nms_batch_1/strided_slice_1StridedSlicenms_batch/batch_pad_1/stack_1!nms_batch_1/strided_slice_1/stack#nms_batch_1/strided_slice_1/stack_1#nms_batch_1/strided_slice_1/stack_2*
T0*
Index0*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
b
nms_batch_1/nms/iou_thresholdConst*
valueB
 *ЭЬЬ>*
dtype0*
_output_shapes
: 
d
nms_batch_1/nms/score_thresholdConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 

nms_batch_1/nms/GreaterGreaternms_batch_1/strided_slice_1nms_batch_1/nms/score_threshold*
T0*#
_output_shapes
:џџџџџџџџџ
k
"nms_batch_1/nms/boolean_mask/ShapeShapenms_batch_1/strided_slice*
_output_shapes
:*
T0
z
0nms_batch_1/nms/boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
|
2nms_batch_1/nms/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
|
2nms_batch_1/nms/boolean_mask/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Њ
*nms_batch_1/nms/boolean_mask/strided_sliceStridedSlice"nms_batch_1/nms/boolean_mask/Shape0nms_batch_1/nms/boolean_mask/strided_slice/stack2nms_batch_1/nms/boolean_mask/strided_slice/stack_12nms_batch_1/nms/boolean_mask/strided_slice/stack_2*
_output_shapes
:*
T0*
Index0
}
3nms_batch_1/nms/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
Ћ
!nms_batch_1/nms/boolean_mask/ProdProd*nms_batch_1/nms/boolean_mask/strided_slice3nms_batch_1/nms/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0
m
$nms_batch_1/nms/boolean_mask/Shape_1Shapenms_batch_1/strided_slice*
T0*
_output_shapes
:
|
2nms_batch_1/nms/boolean_mask/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: 
~
4nms_batch_1/nms/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
~
4nms_batch_1/nms/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ф
,nms_batch_1/nms/boolean_mask/strided_slice_1StridedSlice$nms_batch_1/nms/boolean_mask/Shape_12nms_batch_1/nms/boolean_mask/strided_slice_1/stack4nms_batch_1/nms/boolean_mask/strided_slice_1/stack_14nms_batch_1/nms/boolean_mask/strided_slice_1/stack_2*

begin_mask*
T0*
Index0*
_output_shapes
: 
m
$nms_batch_1/nms/boolean_mask/Shape_2Shapenms_batch_1/strided_slice*
_output_shapes
:*
T0
|
2nms_batch_1/nms/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
valueB:*
dtype0
~
4nms_batch_1/nms/boolean_mask/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
~
4nms_batch_1/nms/boolean_mask/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ф
,nms_batch_1/nms/boolean_mask/strided_slice_2StridedSlice$nms_batch_1/nms/boolean_mask/Shape_22nms_batch_1/nms/boolean_mask/strided_slice_2/stack4nms_batch_1/nms/boolean_mask/strided_slice_2/stack_14nms_batch_1/nms/boolean_mask/strided_slice_2/stack_2*
_output_shapes
:*
T0*
Index0*
end_mask

,nms_batch_1/nms/boolean_mask/concat/values_1Pack!nms_batch_1/nms/boolean_mask/Prod*
T0*
N*
_output_shapes
:
j
(nms_batch_1/nms/boolean_mask/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

#nms_batch_1/nms/boolean_mask/concatConcatV2,nms_batch_1/nms/boolean_mask/strided_slice_1,nms_batch_1/nms/boolean_mask/concat/values_1,nms_batch_1/nms/boolean_mask/strided_slice_2(nms_batch_1/nms/boolean_mask/concat/axis*
T0*
N*
_output_shapes
:
Ё
$nms_batch_1/nms/boolean_mask/ReshapeReshapenms_batch_1/strided_slice#nms_batch_1/nms/boolean_mask/concat*
T0*'
_output_shapes
:џџџџџџџџџ

,nms_batch_1/nms/boolean_mask/Reshape_1/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
І
&nms_batch_1/nms/boolean_mask/Reshape_1Reshapenms_batch_1/nms/Greater,nms_batch_1/nms/boolean_mask/Reshape_1/shape*
T0
*#
_output_shapes
:џџџџџџџџџ
|
"nms_batch_1/nms/boolean_mask/WhereWhere&nms_batch_1/nms/boolean_mask/Reshape_1*'
_output_shapes
:џџџџџџџџџ

$nms_batch_1/nms/boolean_mask/SqueezeSqueeze"nms_batch_1/nms/boolean_mask/Where*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
*
T0	
l
*nms_batch_1/nms/boolean_mask/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ў
%nms_batch_1/nms/boolean_mask/GatherV2GatherV2$nms_batch_1/nms/boolean_mask/Reshape$nms_batch_1/nms/boolean_mask/Squeeze*nms_batch_1/nms/boolean_mask/GatherV2/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0
o
$nms_batch_1/nms/boolean_mask_1/ShapeShapenms_batch_1/strided_slice_1*
T0*
_output_shapes
:
|
2nms_batch_1/nms/boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
~
4nms_batch_1/nms/boolean_mask_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4nms_batch_1/nms/boolean_mask_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Д
,nms_batch_1/nms/boolean_mask_1/strided_sliceStridedSlice$nms_batch_1/nms/boolean_mask_1/Shape2nms_batch_1/nms/boolean_mask_1/strided_slice/stack4nms_batch_1/nms/boolean_mask_1/strided_slice/stack_14nms_batch_1/nms/boolean_mask_1/strided_slice/stack_2*
_output_shapes
:*
T0*
Index0

5nms_batch_1/nms/boolean_mask_1/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:
Б
#nms_batch_1/nms/boolean_mask_1/ProdProd,nms_batch_1/nms/boolean_mask_1/strided_slice5nms_batch_1/nms/boolean_mask_1/Prod/reduction_indices*
T0*
_output_shapes
: 
q
&nms_batch_1/nms/boolean_mask_1/Shape_1Shapenms_batch_1/strided_slice_1*
_output_shapes
:*
T0
~
4nms_batch_1/nms/boolean_mask_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:

6nms_batch_1/nms/boolean_mask_1/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

6nms_batch_1/nms/boolean_mask_1/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ю
.nms_batch_1/nms/boolean_mask_1/strided_slice_1StridedSlice&nms_batch_1/nms/boolean_mask_1/Shape_14nms_batch_1/nms/boolean_mask_1/strided_slice_1/stack6nms_batch_1/nms/boolean_mask_1/strided_slice_1/stack_16nms_batch_1/nms/boolean_mask_1/strided_slice_1/stack_2*
_output_shapes
: *
Index0*
T0*

begin_mask
q
&nms_batch_1/nms/boolean_mask_1/Shape_2Shapenms_batch_1/strided_slice_1*
T0*
_output_shapes
:
~
4nms_batch_1/nms/boolean_mask_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:

6nms_batch_1/nms/boolean_mask_1/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

6nms_batch_1/nms/boolean_mask_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ь
.nms_batch_1/nms/boolean_mask_1/strided_slice_2StridedSlice&nms_batch_1/nms/boolean_mask_1/Shape_24nms_batch_1/nms/boolean_mask_1/strided_slice_2/stack6nms_batch_1/nms/boolean_mask_1/strided_slice_2/stack_16nms_batch_1/nms/boolean_mask_1/strided_slice_2/stack_2*
_output_shapes
: *
T0*
Index0*
end_mask

.nms_batch_1/nms/boolean_mask_1/concat/values_1Pack#nms_batch_1/nms/boolean_mask_1/Prod*
_output_shapes
:*
T0*
N
l
*nms_batch_1/nms/boolean_mask_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

%nms_batch_1/nms/boolean_mask_1/concatConcatV2.nms_batch_1/nms/boolean_mask_1/strided_slice_1.nms_batch_1/nms/boolean_mask_1/concat/values_1.nms_batch_1/nms/boolean_mask_1/strided_slice_2*nms_batch_1/nms/boolean_mask_1/concat/axis*
T0*
N*
_output_shapes
:
Ѓ
&nms_batch_1/nms/boolean_mask_1/ReshapeReshapenms_batch_1/strided_slice_1%nms_batch_1/nms/boolean_mask_1/concat*
T0*#
_output_shapes
:џџџџџџџџџ

.nms_batch_1/nms/boolean_mask_1/Reshape_1/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
Њ
(nms_batch_1/nms/boolean_mask_1/Reshape_1Reshapenms_batch_1/nms/Greater.nms_batch_1/nms/boolean_mask_1/Reshape_1/shape*
T0
*#
_output_shapes
:џџџџџџџџџ

$nms_batch_1/nms/boolean_mask_1/WhereWhere(nms_batch_1/nms/boolean_mask_1/Reshape_1*'
_output_shapes
:џџџџџџџџџ

&nms_batch_1/nms/boolean_mask_1/SqueezeSqueeze$nms_batch_1/nms/boolean_mask_1/Where*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ
n
,nms_batch_1/nms/boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0

'nms_batch_1/nms/boolean_mask_1/GatherV2GatherV2&nms_batch_1/nms/boolean_mask_1/Reshape&nms_batch_1/nms/boolean_mask_1/Squeeze,nms_batch_1/nms/boolean_mask_1/GatherV2/axis*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	
`
nms_batch_1/nms/WhereWherenms_batch_1/nms/Greater*'
_output_shapes
:џџџџџџџџџ
p
nms_batch_1/nms/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:

nms_batch_1/nms/ReshapeReshapenms_batch_1/nms/Wherenms_batch_1/nms/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ
v
3nms_batch_1/nms/NonMaxSuppressionV2/max_output_sizeConst*
_output_shapes
: *
value
B :Ш*
dtype0

#nms_batch_1/nms/NonMaxSuppressionV2NonMaxSuppressionV2%nms_batch_1/nms/boolean_mask/GatherV2'nms_batch_1/nms/boolean_mask_1/GatherV23nms_batch_1/nms/NonMaxSuppressionV2/max_output_sizenms_batch_1/nms/iou_threshold*#
_output_shapes
:џџџџџџџџџ
_
nms_batch_1/nms/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
в
nms_batch_1/nms/GatherV2GatherV2nms_batch_1/nms/Reshape#nms_batch_1/nms/NonMaxSuppressionV2nms_batch_1/nms/GatherV2/axis*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0*
Tparams0	
k
!nms_batch_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
m
#nms_batch_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#nms_batch_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

nms_batch_1/strided_slice_2StridedSlicenms_batch/batch_pad/stack_1!nms_batch_1/strided_slice_2/stack#nms_batch_1/strided_slice_2/stack_1#nms_batch_1/strided_slice_2/stack_2*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask*
Index0*
T0
[
nms_batch_1/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ч
nms_batch_1/GatherV2GatherV2nms_batch_1/strided_slice_2nms_batch_1/nms/GatherV2nms_batch_1/GatherV2/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0
k
!nms_batch_1/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
m
#nms_batch_1/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#nms_batch_1/strided_slice_3/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

nms_batch_1/strided_slice_3StridedSlicenms_batch/batch_pad_1/stack_1!nms_batch_1/strided_slice_3/stack#nms_batch_1/strided_slice_3/stack_1#nms_batch_1/strided_slice_3/stack_2*#
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask*
T0*
Index0
]
nms_batch_1/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ч
nms_batch_1/GatherV2_1GatherV2nms_batch_1/strided_slice_3nms_batch_1/nms/GatherV2nms_batch_1/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ
_
nms_batch_1/batch_pad/ShapeShapenms_batch_1/GatherV2*
T0*
_output_shapes
:
s
)nms_batch_1/batch_pad/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
u
+nms_batch_1/batch_pad/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+nms_batch_1/batch_pad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

#nms_batch_1/batch_pad/strided_sliceStridedSlicenms_batch_1/batch_pad/Shape)nms_batch_1/batch_pad/strided_slice/stack+nms_batch_1/batch_pad/strided_slice/stack_1+nms_batch_1/batch_pad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
v
nms_batch_1/batch_pad/stackPack#nms_batch_1/batch_pad/strided_slice*
N*
_output_shapes
:*
T0
e
nms_batch_1/batch_pad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
nms_batch_1/batch_pad/MaxMaxnms_batch_1/batch_pad/stacknms_batch_1/batch_pad/Const*
T0*
_output_shapes
: 
h
$nms_batch_1/batch_pad/pad_axis/ShapeShapenms_batch_1/GatherV2*
T0*
_output_shapes
:

&nms_batch_1/batch_pad/pad_axis/unstackUnpack$nms_batch_1/batch_pad/pad_axis/Shape*	
num*
T0*
_output_shapes
: : 
f
$nms_batch_1/batch_pad/pad_axis/sub/yConst*
value	B : *
dtype0*
_output_shapes
: 

"nms_batch_1/batch_pad/pad_axis/subSubnms_batch_1/batch_pad/Max$nms_batch_1/batch_pad/pad_axis/sub/y*
T0*
_output_shapes
: 

$nms_batch_1/batch_pad/pad_axis/sub_1Sub"nms_batch_1/batch_pad/pad_axis/sub&nms_batch_1/batch_pad/pad_axis/unstack*
T0*
_output_shapes
: 
j
(nms_batch_1/batch_pad/pad_axis/Maximum/yConst*
value	B : *
dtype0*
_output_shapes
: 
Ђ
&nms_batch_1/batch_pad/pad_axis/MaximumMaximum$nms_batch_1/batch_pad/pad_axis/sub_1(nms_batch_1/batch_pad/pad_axis/Maximum/y*
_output_shapes
: *
T0
u
$nms_batch_1/batch_pad/pad_axis/stackConst*
valueB"        *
dtype0*
_output_shapes
:
j
(nms_batch_1/batch_pad/pad_axis/stack_1/1Const*
value	B : *
dtype0*
_output_shapes
: 
Ў
&nms_batch_1/batch_pad/pad_axis/stack_1Pack&nms_batch_1/batch_pad/pad_axis/Maximum(nms_batch_1/batch_pad/pad_axis/stack_1/1*
T0*
N*
_output_shapes
:
К
&nms_batch_1/batch_pad/pad_axis/stack_2Pack$nms_batch_1/batch_pad/pad_axis/stack&nms_batch_1/batch_pad/pad_axis/stack_1*
T0*

axis*
N*
_output_shapes

:
Ђ
"nms_batch_1/batch_pad/pad_axis/PadPadnms_batch_1/GatherV2&nms_batch_1/batch_pad/pad_axis/stack_2*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
j
(nms_batch_1/batch_pad/pad_axis/stack_3/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ё
&nms_batch_1/batch_pad/pad_axis/stack_3Packnms_batch_1/batch_pad/Max(nms_batch_1/batch_pad/pad_axis/stack_3/1*
_output_shapes
:*
T0*
N
Џ
&nms_batch_1/batch_pad/pad_axis/ReshapeReshape"nms_batch_1/batch_pad/pad_axis/Pad&nms_batch_1/batch_pad/pad_axis/stack_3*
T0*'
_output_shapes
:џџџџџџџџџ

nms_batch_1/batch_pad/stack_1Pack&nms_batch_1/batch_pad/pad_axis/Reshape*
N*+
_output_shapes
:џџџџџџџџџ*
T0
c
nms_batch_1/batch_pad_1/ShapeShapenms_batch_1/GatherV2_1*
T0*
_output_shapes
:
u
+nms_batch_1/batch_pad_1/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
w
-nms_batch_1/batch_pad_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-nms_batch_1/batch_pad_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ѕ
%nms_batch_1/batch_pad_1/strided_sliceStridedSlicenms_batch_1/batch_pad_1/Shape+nms_batch_1/batch_pad_1/strided_slice/stack-nms_batch_1/batch_pad_1/strided_slice/stack_1-nms_batch_1/batch_pad_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
z
nms_batch_1/batch_pad_1/stackPack%nms_batch_1/batch_pad_1/strided_slice*
T0*
N*
_output_shapes
:
g
nms_batch_1/batch_pad_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:

nms_batch_1/batch_pad_1/MaxMaxnms_batch_1/batch_pad_1/stacknms_batch_1/batch_pad_1/Const*
T0*
_output_shapes
: 
l
&nms_batch_1/batch_pad_1/pad_axis/ShapeShapenms_batch_1/GatherV2_1*
T0*
_output_shapes
:

(nms_batch_1/batch_pad_1/pad_axis/unstackUnpack&nms_batch_1/batch_pad_1/pad_axis/Shape*
_output_shapes
: *	
num*
T0
h
&nms_batch_1/batch_pad_1/pad_axis/sub/yConst*
value	B : *
dtype0*
_output_shapes
: 

$nms_batch_1/batch_pad_1/pad_axis/subSubnms_batch_1/batch_pad_1/Max&nms_batch_1/batch_pad_1/pad_axis/sub/y*
T0*
_output_shapes
: 

&nms_batch_1/batch_pad_1/pad_axis/sub_1Sub$nms_batch_1/batch_pad_1/pad_axis/sub(nms_batch_1/batch_pad_1/pad_axis/unstack*
_output_shapes
: *
T0
l
*nms_batch_1/batch_pad_1/pad_axis/Maximum/yConst*
value	B : *
dtype0*
_output_shapes
: 
Ј
(nms_batch_1/batch_pad_1/pad_axis/MaximumMaximum&nms_batch_1/batch_pad_1/pad_axis/sub_1*nms_batch_1/batch_pad_1/pad_axis/Maximum/y*
T0*
_output_shapes
: 
p
&nms_batch_1/batch_pad_1/pad_axis/stackConst*
valueB: *
dtype0*
_output_shapes
:

(nms_batch_1/batch_pad_1/pad_axis/stack_1Pack(nms_batch_1/batch_pad_1/pad_axis/Maximum*
T0*
N*
_output_shapes
:
Р
(nms_batch_1/batch_pad_1/pad_axis/stack_2Pack&nms_batch_1/batch_pad_1/pad_axis/stack(nms_batch_1/batch_pad_1/pad_axis/stack_1*
T0*

axis*
N*
_output_shapes

:

$nms_batch_1/batch_pad_1/pad_axis/PadPadnms_batch_1/GatherV2_1(nms_batch_1/batch_pad_1/pad_axis/stack_2*
T0*#
_output_shapes
:џџџџџџџџџ
{
(nms_batch_1/batch_pad_1/pad_axis/stack_3Packnms_batch_1/batch_pad_1/Max*
T0*
N*
_output_shapes
:
Б
(nms_batch_1/batch_pad_1/pad_axis/ReshapeReshape$nms_batch_1/batch_pad_1/pad_axis/Pad(nms_batch_1/batch_pad_1/pad_axis/stack_3*
T0*#
_output_shapes
:џџџџџџџџџ

nms_batch_1/batch_pad_1/stack_1Pack(nms_batch_1/batch_pad_1/pad_axis/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0*
N
A
ShapeShapeinput_images*
T0*
_output_shapes
:
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
_output_shapes
:*
Index0*
T0
R
ToFloatCaststrided_slice*

SrcT0*
_output_shapes
:*

DstT0
f
strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
h
strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ѓ
strided_slice_1StridedSlicenms_batch_1/batch_pad/stack_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask*
ellipsis_mask*
Index0*
T0
_
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: 
a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
З
strided_slice_2StridedSliceToFloatstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
^
mulMulstrided_slice_1strided_slice_2*
T0*'
_output_shapes
:џџџџџџџџџ
f
strided_slice_3/stackConst*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_3/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_3/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ѓ
strided_slice_3StridedSlicenms_batch_1/batch_pad/stack_1strided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
shrink_axis_mask*
Index0*
T0*
ellipsis_mask*'
_output_shapes
:џџџџџџџџџ
_
strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_4/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
З
strided_slice_4StridedSliceToFloatstrided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
`
mul_1Mulstrided_slice_3strided_slice_4*
T0*'
_output_shapes
:џџџџџџџџџ
f
strided_slice_5/stackConst*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_5/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ѓ
strided_slice_5StridedSlicenms_batch_1/batch_pad/stack_1strided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask*
ellipsis_mask*
T0*
Index0
_
strided_slice_6/stackConst*
dtype0*
_output_shapes
:*
valueB: 
a
strided_slice_6/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
З
strided_slice_6StridedSliceToFloatstrided_slice_6/stackstrided_slice_6/stack_1strided_slice_6/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
`
mul_2Mulstrided_slice_5strided_slice_6*'
_output_shapes
:џџџџџџџџџ*
T0
f
strided_slice_7/stackConst*
_output_shapes
:*
valueB"       *
dtype0
h
strided_slice_7/stack_1Const*
_output_shapes
:*
valueB"       *
dtype0
h
strided_slice_7/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ѓ
strided_slice_7StridedSlicenms_batch_1/batch_pad/stack_1strided_slice_7/stackstrided_slice_7/stack_1strided_slice_7/stack_2*
shrink_axis_mask*
T0*
Index0*
ellipsis_mask*'
_output_shapes
:џџџџџџџџџ
_
strided_slice_8/stackConst*
_output_shapes
:*
valueB:*
dtype0
a
strided_slice_8/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
З
strided_slice_8StridedSliceToFloatstrided_slice_8/stackstrided_slice_8/stack_1strided_slice_8/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
`
mul_3Mulstrided_slice_7strided_slice_8*
T0*'
_output_shapes
:џџџџџџџџџ
{
stackPackmulmul_1mul_2mul_3*
N*+
_output_shapes
:џџџџџџџџџ*
T0*
axisџџџџџџџџџ
N
boxesIdentitystack*+
_output_shapes
:џџџџџџџџџ*
T0
e
scoresIdentitynms_batch_1/batch_pad_1/stack_1*
T0*'
_output_shapes
:џџџџџџџџџ

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_8adc4b2354ee44f9bd645e1ee0f5275a/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
N
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
џ
save/SaveV2/tensor_namesConst"/device:CPU:0*Ѓ
valueBOB-PyramidFusedNet/dem0_log_conv0/BatchNorm/betaB.PyramidFusedNet/dem0_log_conv0/BatchNorm/gammaB4PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_meanB8PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_varianceB0PyramidFusedNet/dem0_log_conv0/depthwise_weightsB0PyramidFusedNet/dem0_log_conv0/pointwise_weightsB-PyramidFusedNet/dem0_log_conv1/BatchNorm/betaB.PyramidFusedNet/dem0_log_conv1/BatchNorm/gammaB4PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_meanB8PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_varianceB0PyramidFusedNet/dem0_log_conv1/depthwise_weightsB0PyramidFusedNet/dem0_log_conv1/pointwise_weightsB-PyramidFusedNet/dem0_reg_conv0/BatchNorm/betaB.PyramidFusedNet/dem0_reg_conv0/BatchNorm/gammaB4PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_meanB8PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_varianceB0PyramidFusedNet/dem0_reg_conv0/depthwise_weightsB0PyramidFusedNet/dem0_reg_conv0/pointwise_weightsB-PyramidFusedNet/dem0_reg_conv1/BatchNorm/betaB.PyramidFusedNet/dem0_reg_conv1/BatchNorm/gammaB4PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_meanB8PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_varianceB0PyramidFusedNet/dem0_reg_conv1/depthwise_weightsB0PyramidFusedNet/dem0_reg_conv1/pointwise_weightsB-PyramidFusedNet/dem1_log_conv0/BatchNorm/betaB.PyramidFusedNet/dem1_log_conv0/BatchNorm/gammaB4PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_meanB8PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_varianceB0PyramidFusedNet/dem1_log_conv0/depthwise_weightsB0PyramidFusedNet/dem1_log_conv0/pointwise_weightsB-PyramidFusedNet/dem1_log_conv1/BatchNorm/betaB.PyramidFusedNet/dem1_log_conv1/BatchNorm/gammaB4PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_meanB8PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_varianceB0PyramidFusedNet/dem1_log_conv1/depthwise_weightsB0PyramidFusedNet/dem1_log_conv1/pointwise_weightsB-PyramidFusedNet/dem1_reg_conv0/BatchNorm/betaB.PyramidFusedNet/dem1_reg_conv0/BatchNorm/gammaB4PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_meanB8PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_varianceB0PyramidFusedNet/dem1_reg_conv0/depthwise_weightsB0PyramidFusedNet/dem1_reg_conv0/pointwise_weightsB-PyramidFusedNet/dem1_reg_conv1/BatchNorm/betaB.PyramidFusedNet/dem1_reg_conv1/BatchNorm/gammaB4PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_meanB8PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_varianceB0PyramidFusedNet/dem1_reg_conv1/depthwise_weightsB0PyramidFusedNet/dem1_reg_conv1/pointwise_weightsB(PyramidFusedNet/fem_conv0/BatchNorm/betaB)PyramidFusedNet/fem_conv0/BatchNorm/gammaB/PyramidFusedNet/fem_conv0/BatchNorm/moving_meanB3PyramidFusedNet/fem_conv0/BatchNorm/moving_varianceB+PyramidFusedNet/fem_conv0/depthwise_weightsB+PyramidFusedNet/fem_conv0/pointwise_weightsB(PyramidFusedNet/fem_conv1/BatchNorm/betaB)PyramidFusedNet/fem_conv1/BatchNorm/gammaB/PyramidFusedNet/fem_conv1/BatchNorm/moving_meanB3PyramidFusedNet/fem_conv1/BatchNorm/moving_varianceB+PyramidFusedNet/fem_conv1/depthwise_weightsB+PyramidFusedNet/fem_conv1/pointwise_weightsB(PyramidFusedNet/fem_conv2/BatchNorm/betaB)PyramidFusedNet/fem_conv2/BatchNorm/gammaB/PyramidFusedNet/fem_conv2/BatchNorm/moving_meanB3PyramidFusedNet/fem_conv2/BatchNorm/moving_varianceB+PyramidFusedNet/fem_conv2/depthwise_weightsB+PyramidFusedNet/fem_conv2/pointwise_weightsB(PyramidFusedNet/fem_conv3/BatchNorm/betaB)PyramidFusedNet/fem_conv3/BatchNorm/gammaB/PyramidFusedNet/fem_conv3/BatchNorm/moving_meanB3PyramidFusedNet/fem_conv3/BatchNorm/moving_varianceB+PyramidFusedNet/fem_conv3/depthwise_weightsB+PyramidFusedNet/fem_conv3/pointwise_weightsB(PyramidFusedNet/fem_conv4/BatchNorm/betaB)PyramidFusedNet/fem_conv4/BatchNorm/gammaB/PyramidFusedNet/fem_conv4/BatchNorm/moving_meanB3PyramidFusedNet/fem_conv4/BatchNorm/moving_varianceB+PyramidFusedNet/fem_conv4/depthwise_weightsB+PyramidFusedNet/fem_conv4/pointwise_weightsBglobal_step*
dtype0*
_output_shapes
:O

save/SaveV2/shape_and_slicesConst"/device:CPU:0*Г
valueЉBІOB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:O
п
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices-PyramidFusedNet/dem0_log_conv0/BatchNorm/beta.PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma4PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean8PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance0PyramidFusedNet/dem0_log_conv0/depthwise_weights0PyramidFusedNet/dem0_log_conv0/pointwise_weights-PyramidFusedNet/dem0_log_conv1/BatchNorm/beta.PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma4PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean8PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance0PyramidFusedNet/dem0_log_conv1/depthwise_weights0PyramidFusedNet/dem0_log_conv1/pointwise_weights-PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta.PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma4PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean8PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance0PyramidFusedNet/dem0_reg_conv0/depthwise_weights0PyramidFusedNet/dem0_reg_conv0/pointwise_weights-PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta.PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma4PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean8PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance0PyramidFusedNet/dem0_reg_conv1/depthwise_weights0PyramidFusedNet/dem0_reg_conv1/pointwise_weights-PyramidFusedNet/dem1_log_conv0/BatchNorm/beta.PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma4PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean8PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance0PyramidFusedNet/dem1_log_conv0/depthwise_weights0PyramidFusedNet/dem1_log_conv0/pointwise_weights-PyramidFusedNet/dem1_log_conv1/BatchNorm/beta.PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma4PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean8PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance0PyramidFusedNet/dem1_log_conv1/depthwise_weights0PyramidFusedNet/dem1_log_conv1/pointwise_weights-PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta.PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma4PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean8PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance0PyramidFusedNet/dem1_reg_conv0/depthwise_weights0PyramidFusedNet/dem1_reg_conv0/pointwise_weights-PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta.PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma4PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean8PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance0PyramidFusedNet/dem1_reg_conv1/depthwise_weights0PyramidFusedNet/dem1_reg_conv1/pointwise_weights(PyramidFusedNet/fem_conv0/BatchNorm/beta)PyramidFusedNet/fem_conv0/BatchNorm/gamma/PyramidFusedNet/fem_conv0/BatchNorm/moving_mean3PyramidFusedNet/fem_conv0/BatchNorm/moving_variance+PyramidFusedNet/fem_conv0/depthwise_weights+PyramidFusedNet/fem_conv0/pointwise_weights(PyramidFusedNet/fem_conv1/BatchNorm/beta)PyramidFusedNet/fem_conv1/BatchNorm/gamma/PyramidFusedNet/fem_conv1/BatchNorm/moving_mean3PyramidFusedNet/fem_conv1/BatchNorm/moving_variance+PyramidFusedNet/fem_conv1/depthwise_weights+PyramidFusedNet/fem_conv1/pointwise_weights(PyramidFusedNet/fem_conv2/BatchNorm/beta)PyramidFusedNet/fem_conv2/BatchNorm/gamma/PyramidFusedNet/fem_conv2/BatchNorm/moving_mean3PyramidFusedNet/fem_conv2/BatchNorm/moving_variance+PyramidFusedNet/fem_conv2/depthwise_weights+PyramidFusedNet/fem_conv2/pointwise_weights(PyramidFusedNet/fem_conv3/BatchNorm/beta)PyramidFusedNet/fem_conv3/BatchNorm/gamma/PyramidFusedNet/fem_conv3/BatchNorm/moving_mean3PyramidFusedNet/fem_conv3/BatchNorm/moving_variance+PyramidFusedNet/fem_conv3/depthwise_weights+PyramidFusedNet/fem_conv3/pointwise_weights(PyramidFusedNet/fem_conv4/BatchNorm/beta)PyramidFusedNet/fem_conv4/BatchNorm/gamma/PyramidFusedNet/fem_conv4/BatchNorm/moving_mean3PyramidFusedNet/fem_conv4/BatchNorm/moving_variance+PyramidFusedNet/fem_conv4/depthwise_weights+PyramidFusedNet/fem_conv4/pointwise_weightsglobal_step"/device:CPU:0*]
dtypesS
Q2O	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0

save/RestoreV2/tensor_namesConst"/device:CPU:0*Ѓ
valueBOB-PyramidFusedNet/dem0_log_conv0/BatchNorm/betaB.PyramidFusedNet/dem0_log_conv0/BatchNorm/gammaB4PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_meanB8PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_varianceB0PyramidFusedNet/dem0_log_conv0/depthwise_weightsB0PyramidFusedNet/dem0_log_conv0/pointwise_weightsB-PyramidFusedNet/dem0_log_conv1/BatchNorm/betaB.PyramidFusedNet/dem0_log_conv1/BatchNorm/gammaB4PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_meanB8PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_varianceB0PyramidFusedNet/dem0_log_conv1/depthwise_weightsB0PyramidFusedNet/dem0_log_conv1/pointwise_weightsB-PyramidFusedNet/dem0_reg_conv0/BatchNorm/betaB.PyramidFusedNet/dem0_reg_conv0/BatchNorm/gammaB4PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_meanB8PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_varianceB0PyramidFusedNet/dem0_reg_conv0/depthwise_weightsB0PyramidFusedNet/dem0_reg_conv0/pointwise_weightsB-PyramidFusedNet/dem0_reg_conv1/BatchNorm/betaB.PyramidFusedNet/dem0_reg_conv1/BatchNorm/gammaB4PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_meanB8PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_varianceB0PyramidFusedNet/dem0_reg_conv1/depthwise_weightsB0PyramidFusedNet/dem0_reg_conv1/pointwise_weightsB-PyramidFusedNet/dem1_log_conv0/BatchNorm/betaB.PyramidFusedNet/dem1_log_conv0/BatchNorm/gammaB4PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_meanB8PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_varianceB0PyramidFusedNet/dem1_log_conv0/depthwise_weightsB0PyramidFusedNet/dem1_log_conv0/pointwise_weightsB-PyramidFusedNet/dem1_log_conv1/BatchNorm/betaB.PyramidFusedNet/dem1_log_conv1/BatchNorm/gammaB4PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_meanB8PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_varianceB0PyramidFusedNet/dem1_log_conv1/depthwise_weightsB0PyramidFusedNet/dem1_log_conv1/pointwise_weightsB-PyramidFusedNet/dem1_reg_conv0/BatchNorm/betaB.PyramidFusedNet/dem1_reg_conv0/BatchNorm/gammaB4PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_meanB8PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_varianceB0PyramidFusedNet/dem1_reg_conv0/depthwise_weightsB0PyramidFusedNet/dem1_reg_conv0/pointwise_weightsB-PyramidFusedNet/dem1_reg_conv1/BatchNorm/betaB.PyramidFusedNet/dem1_reg_conv1/BatchNorm/gammaB4PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_meanB8PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_varianceB0PyramidFusedNet/dem1_reg_conv1/depthwise_weightsB0PyramidFusedNet/dem1_reg_conv1/pointwise_weightsB(PyramidFusedNet/fem_conv0/BatchNorm/betaB)PyramidFusedNet/fem_conv0/BatchNorm/gammaB/PyramidFusedNet/fem_conv0/BatchNorm/moving_meanB3PyramidFusedNet/fem_conv0/BatchNorm/moving_varianceB+PyramidFusedNet/fem_conv0/depthwise_weightsB+PyramidFusedNet/fem_conv0/pointwise_weightsB(PyramidFusedNet/fem_conv1/BatchNorm/betaB)PyramidFusedNet/fem_conv1/BatchNorm/gammaB/PyramidFusedNet/fem_conv1/BatchNorm/moving_meanB3PyramidFusedNet/fem_conv1/BatchNorm/moving_varianceB+PyramidFusedNet/fem_conv1/depthwise_weightsB+PyramidFusedNet/fem_conv1/pointwise_weightsB(PyramidFusedNet/fem_conv2/BatchNorm/betaB)PyramidFusedNet/fem_conv2/BatchNorm/gammaB/PyramidFusedNet/fem_conv2/BatchNorm/moving_meanB3PyramidFusedNet/fem_conv2/BatchNorm/moving_varianceB+PyramidFusedNet/fem_conv2/depthwise_weightsB+PyramidFusedNet/fem_conv2/pointwise_weightsB(PyramidFusedNet/fem_conv3/BatchNorm/betaB)PyramidFusedNet/fem_conv3/BatchNorm/gammaB/PyramidFusedNet/fem_conv3/BatchNorm/moving_meanB3PyramidFusedNet/fem_conv3/BatchNorm/moving_varianceB+PyramidFusedNet/fem_conv3/depthwise_weightsB+PyramidFusedNet/fem_conv3/pointwise_weightsB(PyramidFusedNet/fem_conv4/BatchNorm/betaB)PyramidFusedNet/fem_conv4/BatchNorm/gammaB/PyramidFusedNet/fem_conv4/BatchNorm/moving_meanB3PyramidFusedNet/fem_conv4/BatchNorm/moving_varianceB+PyramidFusedNet/fem_conv4/depthwise_weightsB+PyramidFusedNet/fem_conv4/pointwise_weightsBglobal_step*
dtype0*
_output_shapes
:O

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*Г
valueЉBІOB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:O
Ј
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*]
dtypesS
Q2O	*в
_output_shapesП
М:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Л
save/AssignAssign-PyramidFusedNet/dem0_log_conv0/BatchNorm/betasave/RestoreV2*@
_class6
42loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/beta*
_output_shapes
: *
T0
С
save/Assign_1Assign.PyramidFusedNet/dem0_log_conv0/BatchNorm/gammasave/RestoreV2:1*
T0*A
_class7
53loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma*
_output_shapes
: 
Э
save/Assign_2Assign4PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_meansave/RestoreV2:2*
T0*G
_class=
;9loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean*
_output_shapes
: 
е
save/Assign_3Assign8PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variancesave/RestoreV2:3*K
_classA
?=loc:@PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance*
_output_shapes
: *
T0
в
save/Assign_4Assign0PyramidFusedNet/dem0_log_conv0/depthwise_weightssave/RestoreV2:4*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/depthwise_weights*'
_output_shapes
:
в
save/Assign_5Assign0PyramidFusedNet/dem0_log_conv0/pointwise_weightssave/RestoreV2:5*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv0/pointwise_weights*'
_output_shapes
: *
T0
П
save/Assign_6Assign-PyramidFusedNet/dem0_log_conv1/BatchNorm/betasave/RestoreV2:6*@
_class6
42loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/beta*
_output_shapes
:*
T0
С
save/Assign_7Assign.PyramidFusedNet/dem0_log_conv1/BatchNorm/gammasave/RestoreV2:7*
T0*A
_class7
53loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma*
_output_shapes
:
Э
save/Assign_8Assign4PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_meansave/RestoreV2:8*
T0*G
_class=
;9loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean*
_output_shapes
:
е
save/Assign_9Assign8PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variancesave/RestoreV2:9*
_output_shapes
:*
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance
г
save/Assign_10Assign0PyramidFusedNet/dem0_log_conv1/depthwise_weightssave/RestoreV2:10*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/depthwise_weights*&
_output_shapes
: 
г
save/Assign_11Assign0PyramidFusedNet/dem0_log_conv1/pointwise_weightssave/RestoreV2:11*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_log_conv1/pointwise_weights*&
_output_shapes
:`
С
save/Assign_12Assign-PyramidFusedNet/dem0_reg_conv0/BatchNorm/betasave/RestoreV2:12*
T0*@
_class6
42loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta*
_output_shapes
: 
У
save/Assign_13Assign.PyramidFusedNet/dem0_reg_conv0/BatchNorm/gammasave/RestoreV2:13*
T0*A
_class7
53loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma*
_output_shapes
: 
Я
save/Assign_14Assign4PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_meansave/RestoreV2:14*
T0*G
_class=
;9loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean*
_output_shapes
: 
з
save/Assign_15Assign8PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variancesave/RestoreV2:15*
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance*
_output_shapes
: 
г
save/Assign_16Assign0PyramidFusedNet/dem0_reg_conv0/depthwise_weightssave/RestoreV2:16*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/depthwise_weights
г
save/Assign_17Assign0PyramidFusedNet/dem0_reg_conv0/pointwise_weightssave/RestoreV2:17*&
_output_shapes
:` *
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv0/pointwise_weights
С
save/Assign_18Assign-PyramidFusedNet/dem0_reg_conv1/BatchNorm/betasave/RestoreV2:18*
_output_shapes
:*
T0*@
_class6
42loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta
У
save/Assign_19Assign.PyramidFusedNet/dem0_reg_conv1/BatchNorm/gammasave/RestoreV2:19*A
_class7
53loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma*
_output_shapes
:*
T0
Я
save/Assign_20Assign4PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_meansave/RestoreV2:20*G
_class=
;9loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean*
_output_shapes
:*
T0
з
save/Assign_21Assign8PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variancesave/RestoreV2:21*
_output_shapes
:*
T0*K
_classA
?=loc:@PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance
г
save/Assign_22Assign0PyramidFusedNet/dem0_reg_conv1/depthwise_weightssave/RestoreV2:22*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/depthwise_weights*&
_output_shapes
: 
г
save/Assign_23Assign0PyramidFusedNet/dem0_reg_conv1/pointwise_weightssave/RestoreV2:23*
T0*C
_class9
75loc:@PyramidFusedNet/dem0_reg_conv1/pointwise_weights*&
_output_shapes
:`
С
save/Assign_24Assign-PyramidFusedNet/dem1_log_conv0/BatchNorm/betasave/RestoreV2:24*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/beta*
_output_shapes
: 
У
save/Assign_25Assign.PyramidFusedNet/dem1_log_conv0/BatchNorm/gammasave/RestoreV2:25*A
_class7
53loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma*
_output_shapes
: *
T0
Я
save/Assign_26Assign4PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_meansave/RestoreV2:26*G
_class=
;9loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean*
_output_shapes
: *
T0
з
save/Assign_27Assign8PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variancesave/RestoreV2:27*
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance*
_output_shapes
: 
д
save/Assign_28Assign0PyramidFusedNet/dem1_log_conv0/depthwise_weightssave/RestoreV2:28*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/depthwise_weights*'
_output_shapes
:
д
save/Assign_29Assign0PyramidFusedNet/dem1_log_conv0/pointwise_weightssave/RestoreV2:29*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv0/pointwise_weights*'
_output_shapes
: 
С
save/Assign_30Assign-PyramidFusedNet/dem1_log_conv1/BatchNorm/betasave/RestoreV2:30*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/beta*
_output_shapes
:
У
save/Assign_31Assign.PyramidFusedNet/dem1_log_conv1/BatchNorm/gammasave/RestoreV2:31*
T0*A
_class7
53loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma*
_output_shapes
:
Я
save/Assign_32Assign4PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_meansave/RestoreV2:32*
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean*
_output_shapes
:
з
save/Assign_33Assign8PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variancesave/RestoreV2:33*
_output_shapes
:*
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance
г
save/Assign_34Assign0PyramidFusedNet/dem1_log_conv1/depthwise_weightssave/RestoreV2:34*&
_output_shapes
: *
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/depthwise_weights
г
save/Assign_35Assign0PyramidFusedNet/dem1_log_conv1/pointwise_weightssave/RestoreV2:35*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_log_conv1/pointwise_weights*&
_output_shapes
:`
С
save/Assign_36Assign-PyramidFusedNet/dem1_reg_conv0/BatchNorm/betasave/RestoreV2:36*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta*
_output_shapes
: 
У
save/Assign_37Assign.PyramidFusedNet/dem1_reg_conv0/BatchNorm/gammasave/RestoreV2:37*
T0*A
_class7
53loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma*
_output_shapes
: 
Я
save/Assign_38Assign4PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_meansave/RestoreV2:38*
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean*
_output_shapes
: 
з
save/Assign_39Assign8PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variancesave/RestoreV2:39*
T0*K
_classA
?=loc:@PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance*
_output_shapes
: 
г
save/Assign_40Assign0PyramidFusedNet/dem1_reg_conv0/depthwise_weightssave/RestoreV2:40*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/depthwise_weights*&
_output_shapes
: 
г
save/Assign_41Assign0PyramidFusedNet/dem1_reg_conv0/pointwise_weightssave/RestoreV2:41*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv0/pointwise_weights*&
_output_shapes
:` 
С
save/Assign_42Assign-PyramidFusedNet/dem1_reg_conv1/BatchNorm/betasave/RestoreV2:42*
T0*@
_class6
42loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta*
_output_shapes
:
У
save/Assign_43Assign.PyramidFusedNet/dem1_reg_conv1/BatchNorm/gammasave/RestoreV2:43*
_output_shapes
:*
T0*A
_class7
53loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma
Я
save/Assign_44Assign4PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_meansave/RestoreV2:44*
T0*G
_class=
;9loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean*
_output_shapes
:
з
save/Assign_45Assign8PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variancesave/RestoreV2:45*K
_classA
?=loc:@PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance*
_output_shapes
:*
T0
г
save/Assign_46Assign0PyramidFusedNet/dem1_reg_conv1/depthwise_weightssave/RestoreV2:46*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/depthwise_weights*&
_output_shapes
: 
г
save/Assign_47Assign0PyramidFusedNet/dem1_reg_conv1/pointwise_weightssave/RestoreV2:47*
T0*C
_class9
75loc:@PyramidFusedNet/dem1_reg_conv1/pointwise_weights*&
_output_shapes
:`
З
save/Assign_48Assign(PyramidFusedNet/fem_conv0/BatchNorm/betasave/RestoreV2:48*;
_class1
/-loc:@PyramidFusedNet/fem_conv0/BatchNorm/beta*
_output_shapes
: *
T0
Й
save/Assign_49Assign)PyramidFusedNet/fem_conv0/BatchNorm/gammasave/RestoreV2:49*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv0/BatchNorm/gamma*
_output_shapes
: 
Х
save/Assign_50Assign/PyramidFusedNet/fem_conv0/BatchNorm/moving_meansave/RestoreV2:50*B
_class8
64loc:@PyramidFusedNet/fem_conv0/BatchNorm/moving_mean*
_output_shapes
: *
T0
Э
save/Assign_51Assign3PyramidFusedNet/fem_conv0/BatchNorm/moving_variancesave/RestoreV2:51*F
_class<
:8loc:@PyramidFusedNet/fem_conv0/BatchNorm/moving_variance*
_output_shapes
: *
T0
Щ
save/Assign_52Assign+PyramidFusedNet/fem_conv0/depthwise_weightssave/RestoreV2:52*&
_output_shapes
: *
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/depthwise_weights
Щ
save/Assign_53Assign+PyramidFusedNet/fem_conv0/pointwise_weightssave/RestoreV2:53*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv0/pointwise_weights*&
_output_shapes
:` 
З
save/Assign_54Assign(PyramidFusedNet/fem_conv1/BatchNorm/betasave/RestoreV2:54*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv1/BatchNorm/beta*
_output_shapes
:@
Й
save/Assign_55Assign)PyramidFusedNet/fem_conv1/BatchNorm/gammasave/RestoreV2:55*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv1/BatchNorm/gamma*
_output_shapes
:@
Х
save/Assign_56Assign/PyramidFusedNet/fem_conv1/BatchNorm/moving_meansave/RestoreV2:56*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv1/BatchNorm/moving_mean*
_output_shapes
:@
Э
save/Assign_57Assign3PyramidFusedNet/fem_conv1/BatchNorm/moving_variancesave/RestoreV2:57*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv1/BatchNorm/moving_variance*
_output_shapes
:@
Щ
save/Assign_58Assign+PyramidFusedNet/fem_conv1/depthwise_weightssave/RestoreV2:58*&
_output_shapes
: *
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/depthwise_weights
Щ
save/Assign_59Assign+PyramidFusedNet/fem_conv1/pointwise_weightssave/RestoreV2:59*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv1/pointwise_weights*&
_output_shapes
:`@
З
save/Assign_60Assign(PyramidFusedNet/fem_conv2/BatchNorm/betasave/RestoreV2:60*
_output_shapes
:@*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv2/BatchNorm/beta
Й
save/Assign_61Assign)PyramidFusedNet/fem_conv2/BatchNorm/gammasave/RestoreV2:61*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv2/BatchNorm/gamma*
_output_shapes
:@
Х
save/Assign_62Assign/PyramidFusedNet/fem_conv2/BatchNorm/moving_meansave/RestoreV2:62*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv2/BatchNorm/moving_mean*
_output_shapes
:@
Э
save/Assign_63Assign3PyramidFusedNet/fem_conv2/BatchNorm/moving_variancesave/RestoreV2:63*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv2/BatchNorm/moving_variance*
_output_shapes
:@
Щ
save/Assign_64Assign+PyramidFusedNet/fem_conv2/depthwise_weightssave/RestoreV2:64*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/depthwise_weights*&
_output_shapes
:@
Ъ
save/Assign_65Assign+PyramidFusedNet/fem_conv2/pointwise_weightssave/RestoreV2:65*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv2/pointwise_weights*'
_output_shapes
:Р@
З
save/Assign_66Assign(PyramidFusedNet/fem_conv3/BatchNorm/betasave/RestoreV2:66*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv3/BatchNorm/beta*
_output_shapes
:@
Й
save/Assign_67Assign)PyramidFusedNet/fem_conv3/BatchNorm/gammasave/RestoreV2:67*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv3/BatchNorm/gamma*
_output_shapes
:@
Х
save/Assign_68Assign/PyramidFusedNet/fem_conv3/BatchNorm/moving_meansave/RestoreV2:68*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv3/BatchNorm/moving_mean*
_output_shapes
:@
Э
save/Assign_69Assign3PyramidFusedNet/fem_conv3/BatchNorm/moving_variancesave/RestoreV2:69*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv3/BatchNorm/moving_variance*
_output_shapes
:@
Щ
save/Assign_70Assign+PyramidFusedNet/fem_conv3/depthwise_weightssave/RestoreV2:70*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/depthwise_weights*&
_output_shapes
:@
Ъ
save/Assign_71Assign+PyramidFusedNet/fem_conv3/pointwise_weightssave/RestoreV2:71*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv3/pointwise_weights*'
_output_shapes
:Р@
З
save/Assign_72Assign(PyramidFusedNet/fem_conv4/BatchNorm/betasave/RestoreV2:72*
_output_shapes
:@*
T0*;
_class1
/-loc:@PyramidFusedNet/fem_conv4/BatchNorm/beta
Й
save/Assign_73Assign)PyramidFusedNet/fem_conv4/BatchNorm/gammasave/RestoreV2:73*
T0*<
_class2
0.loc:@PyramidFusedNet/fem_conv4/BatchNorm/gamma*
_output_shapes
:@
Х
save/Assign_74Assign/PyramidFusedNet/fem_conv4/BatchNorm/moving_meansave/RestoreV2:74*
_output_shapes
:@*
T0*B
_class8
64loc:@PyramidFusedNet/fem_conv4/BatchNorm/moving_mean
Э
save/Assign_75Assign3PyramidFusedNet/fem_conv4/BatchNorm/moving_variancesave/RestoreV2:75*
_output_shapes
:@*
T0*F
_class<
:8loc:@PyramidFusedNet/fem_conv4/BatchNorm/moving_variance
Щ
save/Assign_76Assign+PyramidFusedNet/fem_conv4/depthwise_weightssave/RestoreV2:76*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/depthwise_weights*&
_output_shapes
:@
Ъ
save/Assign_77Assign+PyramidFusedNet/fem_conv4/pointwise_weightssave/RestoreV2:77*
T0*>
_class4
20loc:@PyramidFusedNet/fem_conv4/pointwise_weights*'
_output_shapes
:Р@
y
save/Assign_78Assignglobal_stepsave/RestoreV2:78*
T0	*
_class
loc:@global_step*
_output_shapes
: 
Э

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"ђ

cond_contextс
о

Ѕ
&image_pyramid/img_shape/cond/cond_text&image_pyramid/img_shape/cond/pred_id:0'image_pyramid/img_shape/cond/switch_t:0 *Ї
image_pyramid/Identity:0
+image_pyramid/img_shape/cond/Shape/Switch:1
$image_pyramid/img_shape/cond/Shape:0
&image_pyramid/img_shape/cond/pred_id:0
2image_pyramid/img_shape/cond/strided_slice/stack:0
4image_pyramid/img_shape/cond/strided_slice/stack_1:0
4image_pyramid/img_shape/cond/strided_slice/stack_2:0
,image_pyramid/img_shape/cond/strided_slice:0
'image_pyramid/img_shape/cond/switch_t:0G
image_pyramid/Identity:0+image_pyramid/img_shape/cond/Shape/Switch:1P
&image_pyramid/img_shape/cond/pred_id:0&image_pyramid/img_shape/cond/pred_id:0
Г
(image_pyramid/img_shape/cond/cond_text_1&image_pyramid/img_shape/cond/pred_id:0'image_pyramid/img_shape/cond/switch_f:0*Е
image_pyramid/Identity:0
-image_pyramid/img_shape/cond/Shape_1/Switch:0
&image_pyramid/img_shape/cond/Shape_1:0
&image_pyramid/img_shape/cond/pred_id:0
4image_pyramid/img_shape/cond/strided_slice_1/stack:0
6image_pyramid/img_shape/cond/strided_slice_1/stack_1:0
6image_pyramid/img_shape/cond/strided_slice_1/stack_2:0
.image_pyramid/img_shape/cond/strided_slice_1:0
'image_pyramid/img_shape/cond/switch_f:0I
image_pyramid/Identity:0-image_pyramid/img_shape/cond/Shape_1/Switch:0P
&image_pyramid/img_shape/cond/pred_id:0&image_pyramid/img_shape/cond/pred_id:0"%
saved_model_main_op


group_deps"З
model_variablesЂ
у
-PyramidFusedNet/fem_conv0/depthwise_weights:02PyramidFusedNet/fem_conv0/depthwise_weights/Assign2PyramidFusedNet/fem_conv0/depthwise_weights/read:02HPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv0/pointwise_weights:02PyramidFusedNet/fem_conv0/pointwise_weights/Assign2PyramidFusedNet/fem_conv0/pointwise_weights/read:02HPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv0/BatchNorm/gamma:00PyramidFusedNet/fem_conv0/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv0/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv0/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv0/BatchNorm/beta:0/PyramidFusedNet/fem_conv0/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv0/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv0/BatchNorm/beta/Initializer/zeros:08
ш
1PyramidFusedNet/fem_conv0/BatchNorm/moving_mean:06PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/Assign6PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/read:02CPyramidFusedNet/fem_conv0/BatchNorm/moving_mean/Initializer/zeros:0
ї
5PyramidFusedNet/fem_conv0/BatchNorm/moving_variance:0:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/Assign:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/read:02FPyramidFusedNet/fem_conv0/BatchNorm/moving_variance/Initializer/ones:0
у
-PyramidFusedNet/fem_conv1/depthwise_weights:02PyramidFusedNet/fem_conv1/depthwise_weights/Assign2PyramidFusedNet/fem_conv1/depthwise_weights/read:02HPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv1/pointwise_weights:02PyramidFusedNet/fem_conv1/pointwise_weights/Assign2PyramidFusedNet/fem_conv1/pointwise_weights/read:02HPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv1/BatchNorm/gamma:00PyramidFusedNet/fem_conv1/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv1/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv1/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv1/BatchNorm/beta:0/PyramidFusedNet/fem_conv1/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv1/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv1/BatchNorm/beta/Initializer/zeros:08
ш
1PyramidFusedNet/fem_conv1/BatchNorm/moving_mean:06PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/Assign6PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/read:02CPyramidFusedNet/fem_conv1/BatchNorm/moving_mean/Initializer/zeros:0
ї
5PyramidFusedNet/fem_conv1/BatchNorm/moving_variance:0:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/Assign:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/read:02FPyramidFusedNet/fem_conv1/BatchNorm/moving_variance/Initializer/ones:0
у
-PyramidFusedNet/fem_conv2/depthwise_weights:02PyramidFusedNet/fem_conv2/depthwise_weights/Assign2PyramidFusedNet/fem_conv2/depthwise_weights/read:02HPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv2/pointwise_weights:02PyramidFusedNet/fem_conv2/pointwise_weights/Assign2PyramidFusedNet/fem_conv2/pointwise_weights/read:02HPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv2/BatchNorm/gamma:00PyramidFusedNet/fem_conv2/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv2/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv2/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv2/BatchNorm/beta:0/PyramidFusedNet/fem_conv2/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv2/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv2/BatchNorm/beta/Initializer/zeros:08
ш
1PyramidFusedNet/fem_conv2/BatchNorm/moving_mean:06PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/Assign6PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/read:02CPyramidFusedNet/fem_conv2/BatchNorm/moving_mean/Initializer/zeros:0
ї
5PyramidFusedNet/fem_conv2/BatchNorm/moving_variance:0:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/Assign:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/read:02FPyramidFusedNet/fem_conv2/BatchNorm/moving_variance/Initializer/ones:0
у
-PyramidFusedNet/fem_conv3/depthwise_weights:02PyramidFusedNet/fem_conv3/depthwise_weights/Assign2PyramidFusedNet/fem_conv3/depthwise_weights/read:02HPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv3/pointwise_weights:02PyramidFusedNet/fem_conv3/pointwise_weights/Assign2PyramidFusedNet/fem_conv3/pointwise_weights/read:02HPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv3/BatchNorm/gamma:00PyramidFusedNet/fem_conv3/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv3/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv3/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv3/BatchNorm/beta:0/PyramidFusedNet/fem_conv3/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv3/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv3/BatchNorm/beta/Initializer/zeros:08
ш
1PyramidFusedNet/fem_conv3/BatchNorm/moving_mean:06PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/Assign6PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/read:02CPyramidFusedNet/fem_conv3/BatchNorm/moving_mean/Initializer/zeros:0
ї
5PyramidFusedNet/fem_conv3/BatchNorm/moving_variance:0:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/Assign:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/read:02FPyramidFusedNet/fem_conv3/BatchNorm/moving_variance/Initializer/ones:0
у
-PyramidFusedNet/fem_conv4/depthwise_weights:02PyramidFusedNet/fem_conv4/depthwise_weights/Assign2PyramidFusedNet/fem_conv4/depthwise_weights/read:02HPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv4/pointwise_weights:02PyramidFusedNet/fem_conv4/pointwise_weights/Assign2PyramidFusedNet/fem_conv4/pointwise_weights/read:02HPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv4/BatchNorm/gamma:00PyramidFusedNet/fem_conv4/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv4/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv4/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv4/BatchNorm/beta:0/PyramidFusedNet/fem_conv4/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv4/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv4/BatchNorm/beta/Initializer/zeros:08
ш
1PyramidFusedNet/fem_conv4/BatchNorm/moving_mean:06PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/Assign6PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/read:02CPyramidFusedNet/fem_conv4/BatchNorm/moving_mean/Initializer/zeros:0
ї
5PyramidFusedNet/fem_conv4/BatchNorm/moving_variance:0:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/Assign:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/read:02FPyramidFusedNet/fem_conv4/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem0_log_conv0/depthwise_weights:07PyramidFusedNet/dem0_log_conv0/depthwise_weights/Assign7PyramidFusedNet/dem0_log_conv0/depthwise_weights/read:02MPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_log_conv0/pointwise_weights:07PyramidFusedNet/dem0_log_conv0/pointwise_weights/Assign7PyramidFusedNet/dem0_log_conv0/pointwise_weights/read:02MPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma:05PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_log_conv0/BatchNorm/beta:04PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem0_log_conv0/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean:0;PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance:0?PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem0_log_conv1/depthwise_weights:07PyramidFusedNet/dem0_log_conv1/depthwise_weights/Assign7PyramidFusedNet/dem0_log_conv1/depthwise_weights/read:02MPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_log_conv1/pointwise_weights:07PyramidFusedNet/dem0_log_conv1/pointwise_weights/Assign7PyramidFusedNet/dem0_log_conv1/pointwise_weights/read:02MPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma:05PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_log_conv1/BatchNorm/beta:04PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem0_log_conv1/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean:0;PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance:0?PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem0_reg_conv0/depthwise_weights:07PyramidFusedNet/dem0_reg_conv0/depthwise_weights/Assign7PyramidFusedNet/dem0_reg_conv0/depthwise_weights/read:02MPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_reg_conv0/pointwise_weights:07PyramidFusedNet/dem0_reg_conv0/pointwise_weights/Assign7PyramidFusedNet/dem0_reg_conv0/pointwise_weights/read:02MPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma:05PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta:04PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean:0;PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance:0?PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem0_reg_conv1/depthwise_weights:07PyramidFusedNet/dem0_reg_conv1/depthwise_weights/Assign7PyramidFusedNet/dem0_reg_conv1/depthwise_weights/read:02MPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_reg_conv1/pointwise_weights:07PyramidFusedNet/dem0_reg_conv1/pointwise_weights/Assign7PyramidFusedNet/dem0_reg_conv1/pointwise_weights/read:02MPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma:05PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta:04PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean:0;PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance:0?PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem1_log_conv0/depthwise_weights:07PyramidFusedNet/dem1_log_conv0/depthwise_weights/Assign7PyramidFusedNet/dem1_log_conv0/depthwise_weights/read:02MPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_log_conv0/pointwise_weights:07PyramidFusedNet/dem1_log_conv0/pointwise_weights/Assign7PyramidFusedNet/dem1_log_conv0/pointwise_weights/read:02MPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma:05PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_log_conv0/BatchNorm/beta:04PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem1_log_conv0/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean:0;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance:0?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem1_log_conv1/depthwise_weights:07PyramidFusedNet/dem1_log_conv1/depthwise_weights/Assign7PyramidFusedNet/dem1_log_conv1/depthwise_weights/read:02MPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_log_conv1/pointwise_weights:07PyramidFusedNet/dem1_log_conv1/pointwise_weights/Assign7PyramidFusedNet/dem1_log_conv1/pointwise_weights/read:02MPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma:05PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_log_conv1/BatchNorm/beta:04PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem1_log_conv1/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean:0;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance:0?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem1_reg_conv0/depthwise_weights:07PyramidFusedNet/dem1_reg_conv0/depthwise_weights/Assign7PyramidFusedNet/dem1_reg_conv0/depthwise_weights/read:02MPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_reg_conv0/pointwise_weights:07PyramidFusedNet/dem1_reg_conv0/pointwise_weights/Assign7PyramidFusedNet/dem1_reg_conv0/pointwise_weights/read:02MPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma:05PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta:04PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean:0;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance:0?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem1_reg_conv1/depthwise_weights:07PyramidFusedNet/dem1_reg_conv1/depthwise_weights/Assign7PyramidFusedNet/dem1_reg_conv1/depthwise_weights/read:02MPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_reg_conv1/pointwise_weights:07PyramidFusedNet/dem1_reg_conv1/pointwise_weights/Assign7PyramidFusedNet/dem1_reg_conv1/pointwise_weights/read:02MPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma:05PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta:04PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean:0;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance:0?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/Initializer/ones:0"и^
trainable_variablesР^Н^
у
-PyramidFusedNet/fem_conv0/depthwise_weights:02PyramidFusedNet/fem_conv0/depthwise_weights/Assign2PyramidFusedNet/fem_conv0/depthwise_weights/read:02HPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv0/pointwise_weights:02PyramidFusedNet/fem_conv0/pointwise_weights/Assign2PyramidFusedNet/fem_conv0/pointwise_weights/read:02HPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv0/BatchNorm/gamma:00PyramidFusedNet/fem_conv0/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv0/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv0/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv0/BatchNorm/beta:0/PyramidFusedNet/fem_conv0/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv0/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv0/BatchNorm/beta/Initializer/zeros:08
у
-PyramidFusedNet/fem_conv1/depthwise_weights:02PyramidFusedNet/fem_conv1/depthwise_weights/Assign2PyramidFusedNet/fem_conv1/depthwise_weights/read:02HPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv1/pointwise_weights:02PyramidFusedNet/fem_conv1/pointwise_weights/Assign2PyramidFusedNet/fem_conv1/pointwise_weights/read:02HPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv1/BatchNorm/gamma:00PyramidFusedNet/fem_conv1/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv1/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv1/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv1/BatchNorm/beta:0/PyramidFusedNet/fem_conv1/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv1/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv1/BatchNorm/beta/Initializer/zeros:08
у
-PyramidFusedNet/fem_conv2/depthwise_weights:02PyramidFusedNet/fem_conv2/depthwise_weights/Assign2PyramidFusedNet/fem_conv2/depthwise_weights/read:02HPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv2/pointwise_weights:02PyramidFusedNet/fem_conv2/pointwise_weights/Assign2PyramidFusedNet/fem_conv2/pointwise_weights/read:02HPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv2/BatchNorm/gamma:00PyramidFusedNet/fem_conv2/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv2/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv2/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv2/BatchNorm/beta:0/PyramidFusedNet/fem_conv2/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv2/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv2/BatchNorm/beta/Initializer/zeros:08
у
-PyramidFusedNet/fem_conv3/depthwise_weights:02PyramidFusedNet/fem_conv3/depthwise_weights/Assign2PyramidFusedNet/fem_conv3/depthwise_weights/read:02HPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv3/pointwise_weights:02PyramidFusedNet/fem_conv3/pointwise_weights/Assign2PyramidFusedNet/fem_conv3/pointwise_weights/read:02HPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv3/BatchNorm/gamma:00PyramidFusedNet/fem_conv3/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv3/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv3/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv3/BatchNorm/beta:0/PyramidFusedNet/fem_conv3/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv3/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv3/BatchNorm/beta/Initializer/zeros:08
у
-PyramidFusedNet/fem_conv4/depthwise_weights:02PyramidFusedNet/fem_conv4/depthwise_weights/Assign2PyramidFusedNet/fem_conv4/depthwise_weights/read:02HPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv4/pointwise_weights:02PyramidFusedNet/fem_conv4/pointwise_weights/Assign2PyramidFusedNet/fem_conv4/pointwise_weights/read:02HPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv4/BatchNorm/gamma:00PyramidFusedNet/fem_conv4/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv4/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv4/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv4/BatchNorm/beta:0/PyramidFusedNet/fem_conv4/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv4/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv4/BatchNorm/beta/Initializer/zeros:08
ї
2PyramidFusedNet/dem0_log_conv0/depthwise_weights:07PyramidFusedNet/dem0_log_conv0/depthwise_weights/Assign7PyramidFusedNet/dem0_log_conv0/depthwise_weights/read:02MPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_log_conv0/pointwise_weights:07PyramidFusedNet/dem0_log_conv0/pointwise_weights/Assign7PyramidFusedNet/dem0_log_conv0/pointwise_weights/read:02MPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma:05PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_log_conv0/BatchNorm/beta:04PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem0_log_conv0/BatchNorm/beta/Initializer/zeros:08
ї
2PyramidFusedNet/dem0_log_conv1/depthwise_weights:07PyramidFusedNet/dem0_log_conv1/depthwise_weights/Assign7PyramidFusedNet/dem0_log_conv1/depthwise_weights/read:02MPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_log_conv1/pointwise_weights:07PyramidFusedNet/dem0_log_conv1/pointwise_weights/Assign7PyramidFusedNet/dem0_log_conv1/pointwise_weights/read:02MPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma:05PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_log_conv1/BatchNorm/beta:04PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem0_log_conv1/BatchNorm/beta/Initializer/zeros:08
ї
2PyramidFusedNet/dem0_reg_conv0/depthwise_weights:07PyramidFusedNet/dem0_reg_conv0/depthwise_weights/Assign7PyramidFusedNet/dem0_reg_conv0/depthwise_weights/read:02MPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_reg_conv0/pointwise_weights:07PyramidFusedNet/dem0_reg_conv0/pointwise_weights/Assign7PyramidFusedNet/dem0_reg_conv0/pointwise_weights/read:02MPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma:05PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta:04PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/Initializer/zeros:08
ї
2PyramidFusedNet/dem0_reg_conv1/depthwise_weights:07PyramidFusedNet/dem0_reg_conv1/depthwise_weights/Assign7PyramidFusedNet/dem0_reg_conv1/depthwise_weights/read:02MPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_reg_conv1/pointwise_weights:07PyramidFusedNet/dem0_reg_conv1/pointwise_weights/Assign7PyramidFusedNet/dem0_reg_conv1/pointwise_weights/read:02MPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma:05PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta:04PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/Initializer/zeros:08
ї
2PyramidFusedNet/dem1_log_conv0/depthwise_weights:07PyramidFusedNet/dem1_log_conv0/depthwise_weights/Assign7PyramidFusedNet/dem1_log_conv0/depthwise_weights/read:02MPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_log_conv0/pointwise_weights:07PyramidFusedNet/dem1_log_conv0/pointwise_weights/Assign7PyramidFusedNet/dem1_log_conv0/pointwise_weights/read:02MPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma:05PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_log_conv0/BatchNorm/beta:04PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem1_log_conv0/BatchNorm/beta/Initializer/zeros:08
ї
2PyramidFusedNet/dem1_log_conv1/depthwise_weights:07PyramidFusedNet/dem1_log_conv1/depthwise_weights/Assign7PyramidFusedNet/dem1_log_conv1/depthwise_weights/read:02MPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_log_conv1/pointwise_weights:07PyramidFusedNet/dem1_log_conv1/pointwise_weights/Assign7PyramidFusedNet/dem1_log_conv1/pointwise_weights/read:02MPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma:05PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_log_conv1/BatchNorm/beta:04PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem1_log_conv1/BatchNorm/beta/Initializer/zeros:08
ї
2PyramidFusedNet/dem1_reg_conv0/depthwise_weights:07PyramidFusedNet/dem1_reg_conv0/depthwise_weights/Assign7PyramidFusedNet/dem1_reg_conv0/depthwise_weights/read:02MPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_reg_conv0/pointwise_weights:07PyramidFusedNet/dem1_reg_conv0/pointwise_weights/Assign7PyramidFusedNet/dem1_reg_conv0/pointwise_weights/read:02MPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma:05PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta:04PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/Initializer/zeros:08
ї
2PyramidFusedNet/dem1_reg_conv1/depthwise_weights:07PyramidFusedNet/dem1_reg_conv1/depthwise_weights/Assign7PyramidFusedNet/dem1_reg_conv1/depthwise_weights/read:02MPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_reg_conv1/pointwise_weights:07PyramidFusedNet/dem1_reg_conv1/pointwise_weights/Assign7PyramidFusedNet/dem1_reg_conv1/pointwise_weights/read:02MPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma:05PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta:04PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/Initializer/zeros:08"чо
while_contextдоао
ф
!image_pyramid/while/while_context*image_pyramid/while/LoopCond:02image_pyramid/while/Merge:0:image_pyramid/while/Identity:0Bimage_pyramid/while/Exit:0Bimage_pyramid/while/Exit_1:0Bimage_pyramid/while/Exit_2:0JЊ
image_pyramid/Identity:0
image_pyramid/TensorArray:0
image_pyramid/ToFloat:0
image_pyramid/while/Ceil:0
image_pyramid/while/Const_1:0
image_pyramid/while/Enter:0
image_pyramid/while/Enter_1:0
image_pyramid/while/Enter_2:0
image_pyramid/while/Exit:0
image_pyramid/while/Exit_1:0
image_pyramid/while/Exit_2:0
image_pyramid/while/Greater/y:0
image_pyramid/while/Greater:0
image_pyramid/while/Identity:0
 image_pyramid/while/Identity_1:0
 image_pyramid/while/Identity_2:0
image_pyramid/while/LoopCond:0
image_pyramid/while/Merge:0
image_pyramid/while/Merge:1
image_pyramid/while/Merge_1:0
image_pyramid/while/Merge_1:1
image_pyramid/while/Merge_2:0
image_pyramid/while/Merge_2:1
image_pyramid/while/Min:0
#image_pyramid/while/NextIteration:0
%image_pyramid/while/NextIteration_1:0
%image_pyramid/while/NextIteration_2:0
*image_pyramid/while/ResizeBilinear/Enter:0
$image_pyramid/while/ResizeBilinear:0
image_pyramid/while/Switch:0
image_pyramid/while/Switch:1
image_pyramid/while/Switch_1:0
image_pyramid/while/Switch_1:1
image_pyramid/while/Switch_2:0
image_pyramid/while/Switch_2:1
?image_pyramid/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
9image_pyramid/while/TensorArrayWrite/TensorArrayWriteV3:0
image_pyramid/while/ToInt32:0
image_pyramid/while/add/y:0
image_pyramid/while/add:0
image_pyramid/while/mul:0
image_pyramid/while/mul_1/y:0
image_pyramid/while/mul_1:0
#image_pyramid/while/truediv/Enter:0
image_pyramid/while/truediv:0F
image_pyramid/Identity:0*image_pyramid/while/ResizeBilinear/Enter:0^
image_pyramid/TensorArray:0?image_pyramid/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0>
image_pyramid/ToFloat:0#image_pyramid/while/truediv/Enter:0Rimage_pyramid/while/Enter:0Rimage_pyramid/while/Enter_1:0Rimage_pyramid/while/Enter_2:0
хЮ
while/while_context*while/LoopCond:02while/Merge:0:while/Identity:0Bwhile/Exit:0Bwhile/Exit_1:0Bwhile/Exit_2:0JЂ
4PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/read:0
5PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/read:0
;PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem0_log_conv0/depthwise_weights/read:0
7PyramidFusedNet/dem0_log_conv0/pointwise_weights/read:0
4PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/read:0
5PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/read:0
;PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem0_log_conv1/depthwise_weights/read:0
7PyramidFusedNet/dem0_log_conv1/pointwise_weights/read:0
4PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/read:0
5PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/read:0
;PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem0_reg_conv0/depthwise_weights/read:0
7PyramidFusedNet/dem0_reg_conv0/pointwise_weights/read:0
4PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/read:0
5PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/read:0
;PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem0_reg_conv1/depthwise_weights/read:0
7PyramidFusedNet/dem0_reg_conv1/pointwise_weights/read:0
4PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_log_conv0/depthwise_weights/read:0
7PyramidFusedNet/dem1_log_conv0/pointwise_weights/read:0
4PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_log_conv1/depthwise_weights/read:0
7PyramidFusedNet/dem1_log_conv1/pointwise_weights/read:0
4PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_reg_conv0/depthwise_weights/read:0
7PyramidFusedNet/dem1_reg_conv0/pointwise_weights/read:0
4PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_reg_conv1/depthwise_weights/read:0
7PyramidFusedNet/dem1_reg_conv1/pointwise_weights/read:0
/PyramidFusedNet/fem_conv0/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv0/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv0/depthwise_weights/read:0
2PyramidFusedNet/fem_conv0/pointwise_weights/read:0
/PyramidFusedNet/fem_conv1/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv1/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv1/depthwise_weights/read:0
2PyramidFusedNet/fem_conv1/pointwise_weights/read:0
/PyramidFusedNet/fem_conv2/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv2/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv2/depthwise_weights/read:0
2PyramidFusedNet/fem_conv2/pointwise_weights/read:0
/PyramidFusedNet/fem_conv3/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv3/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv3/depthwise_weights/read:0
2PyramidFusedNet/fem_conv3/pointwise_weights/read:0
/PyramidFusedNet/fem_conv4/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv4/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv4/depthwise_weights/read:0
2PyramidFusedNet/fem_conv4/pointwise_weights/read:0
image_pyramid/TensorArray:0
image_pyramid/while/Exit_1:0
while/Cast:0
while/Cast_1:0
while/Cast_2:0
while/Cast_3:0
while/Cast_4:0
while/Ceil:0
while/Ceil_1:0
while/Ceil_2:0
while/Ceil_3:0
while/Enter:0
while/Enter_1:0
while/Enter_2:0
while/Equal/y:0
while/Equal:0
while/Exit:0
while/Exit_1:0
while/Exit_2:0
while/Identity:0
while/Identity_1:0
while/Identity_2:0
while/Less:0
while/LoopCond:0
while/Merge:0
while/Merge:1
while/Merge_1:0
while/Merge_1:1
while/Merge_2:0
while/Merge_2:1
while/NextIteration:0
while/NextIteration_1:0
while/NextIteration_2:0
while/Reshape/shape:0
while/Reshape:0
while/Reshape_1/shape:0
while/Reshape_1:0
while/ResizeBilinear:0
while/Switch:0
while/Switch:1
while/Switch_1:0
while/Switch_1:1
while/Switch_2:0
while/Switch_2:1
while/TensorArrayReadV3:0
while/TensorArraySizeV3/Enter:0
!while/TensorArraySizeV3/Enter_1:0
while/TensorArraySizeV3:0
while/ToInt32:0
while/ToInt32_1:0
while/ToInt32_2:0
while/add/y:0
while/add:0
while/add_1/y:0
while/add_1:0
while/add_2/y:0
while/add_2:0
while/add_3/y:0
while/add_3:0
while/add_4/y:0
while/add_4:0
while/add_5/y:0
while/add_5:0
while/concat/axis:0
while/concat:0
while/concat_1/axis:0
while/concat_1:0
while/cond/Merge:0
while/cond/Merge:1
while/cond/Merge_1:0
while/cond/Merge_1:1
6while/cond/PyramidFusedNet/fem_conv0/BatchNorm/Const:0
Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
?while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm:0
?while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm:1
?while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm:2
?while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm:3
?while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm:4
6while/cond/PyramidFusedNet/fem_conv0/LeakyRelu/alpha:0
4while/cond/PyramidFusedNet/fem_conv0/LeakyRelu/mul:0
0while/cond/PyramidFusedNet/fem_conv0/LeakyRelu:0
=while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enter:0
=while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Shape:0
Gwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enter:0
Awhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise:0
Ewhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/dilation_rate:0
7while/cond/PyramidFusedNet/fem_conv0/separable_conv2d:0
6while/cond/PyramidFusedNet/fem_conv1/BatchNorm/Const:0
Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
?while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm:0
?while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm:1
?while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm:2
?while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm:3
?while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm:4
6while/cond/PyramidFusedNet/fem_conv1/LeakyRelu/alpha:0
4while/cond/PyramidFusedNet/fem_conv1/LeakyRelu/mul:0
0while/cond/PyramidFusedNet/fem_conv1/LeakyRelu:0
=while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enter:0
=while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Shape:0
Gwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enter:0
Awhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise:0
Ewhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/dilation_rate:0
7while/cond/PyramidFusedNet/fem_conv1/separable_conv2d:0
6while/cond/PyramidFusedNet/fem_conv2/BatchNorm/Const:0
Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3:0
?while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm:0
?while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm:1
?while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm:2
?while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm:3
?while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm:4
6while/cond/PyramidFusedNet/fem_conv2/LeakyRelu/alpha:0
4while/cond/PyramidFusedNet/fem_conv2/LeakyRelu/mul:0
0while/cond/PyramidFusedNet/fem_conv2/LeakyRelu:0
=while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enter:0
=while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Shape:0
Gwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enter:0
Awhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise:0
Ewhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/dilation_rate:0
7while/cond/PyramidFusedNet/fem_conv2/separable_conv2d:0
6while/cond/PyramidFusedNet/fem_conv3/BatchNorm/Const:0
Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3:0
?while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm:0
?while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm:1
?while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm:2
?while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm:3
?while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm:4
6while/cond/PyramidFusedNet/fem_conv3/LeakyRelu/alpha:0
4while/cond/PyramidFusedNet/fem_conv3/LeakyRelu/mul:0
0while/cond/PyramidFusedNet/fem_conv3/LeakyRelu:0
=while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enter:0
=while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Shape:0
Gwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enter:0
Awhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise:0
Ewhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/dilation_rate:0
7while/cond/PyramidFusedNet/fem_conv3/separable_conv2d:0
6while/cond/PyramidFusedNet/fem_conv4/BatchNorm/Const:0
Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3:0
?while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm:0
?while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm:1
?while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm:2
?while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm:3
?while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm:4
6while/cond/PyramidFusedNet/fem_conv4/LeakyRelu/alpha:0
4while/cond/PyramidFusedNet/fem_conv4/LeakyRelu/mul:0
0while/cond/PyramidFusedNet/fem_conv4/LeakyRelu:0
=while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enter:0
=while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Shape:0
Gwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enter:0
Awhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise:0
Ewhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/dilation_rate:0
7while/cond/PyramidFusedNet/fem_conv4/separable_conv2d:0
8while/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_1/fem_conv0/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_1/fem_conv0/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_1/fem_conv0/LeakyRelu:0
?while/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d:0
8while/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_1/fem_conv1/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_1/fem_conv1/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_1/fem_conv1/LeakyRelu:0
?while/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d:0
8while/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_1/fem_conv2/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_1/fem_conv2/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_1/fem_conv2/LeakyRelu:0
?while/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d:0
8while/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_1/fem_conv3/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_1/fem_conv3/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_1/fem_conv3/LeakyRelu:0
?while/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d:0
8while/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_1/fem_conv4/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_1/fem_conv4/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_1/fem_conv4/LeakyRelu:0
?while/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d:0
=while/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Shape:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Shape:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d:0
=while/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Shape:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Shape:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d:0
4while/cond/PyramidFusedNet_2/softmax/Reshape/shape:0
.while/cond/PyramidFusedNet_2/softmax/Reshape:0
0while/cond/PyramidFusedNet_2/softmax/Reshape_1:0
,while/cond/PyramidFusedNet_2/softmax/Shape:0
.while/cond/PyramidFusedNet_2/softmax/Softmax:0
2while/cond/PyramidFusedNet_2/strided_slice/stack:0
4while/cond/PyramidFusedNet_2/strided_slice/stack_1:0
4while/cond/PyramidFusedNet_2/strided_slice/stack_2:0
,while/cond/PyramidFusedNet_2/strided_slice:0
=while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Shape:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Shape:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d:0
=while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Shape:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Shape:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d:0
4while/cond/PyramidFusedNet_3/softmax/Reshape/shape:0
.while/cond/PyramidFusedNet_3/softmax/Reshape:0
0while/cond/PyramidFusedNet_3/softmax/Reshape_1:0
,while/cond/PyramidFusedNet_3/softmax/Shape:0
.while/cond/PyramidFusedNet_3/softmax/Softmax:0
2while/cond/PyramidFusedNet_3/strided_slice/stack:0
4while/cond/PyramidFusedNet_3/strided_slice/stack_1:0
4while/cond/PyramidFusedNet_3/strided_slice/stack_2:0
,while/cond/PyramidFusedNet_3/strided_slice:0
8while/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_4/fem_conv0/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_4/fem_conv0/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_4/fem_conv0/LeakyRelu:0
?while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d:0
8while/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_4/fem_conv1/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_4/fem_conv1/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_4/fem_conv1/LeakyRelu:0
?while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d:0
8while/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_4/fem_conv2/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_4/fem_conv2/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_4/fem_conv2/LeakyRelu:0
?while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d:0
8while/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_4/fem_conv3/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_4/fem_conv3/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_4/fem_conv3/LeakyRelu:0
?while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d:0
8while/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu:0
?while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d:0
8while/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_5/fem_conv0/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_5/fem_conv0/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_5/fem_conv0/LeakyRelu:0
?while/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d:0
8while/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_5/fem_conv1/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_5/fem_conv1/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_5/fem_conv1/LeakyRelu:0
?while/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d:0
8while/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_5/fem_conv2/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_5/fem_conv2/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_5/fem_conv2/LeakyRelu:0
?while/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d:0
8while/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_5/fem_conv3/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_5/fem_conv3/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_5/fem_conv3/LeakyRelu:0
?while/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d:0
8while/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_5/fem_conv4/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_5/fem_conv4/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_5/fem_conv4/LeakyRelu:0
?while/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d:0
=while/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/Const:0
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/Shape:0
Hwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/Const:0
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/Shape:0
Hwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d:0
=while/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/Const:0
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/Shape:0
Hwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/Const:0
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/Shape:0
Hwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d:0
4while/cond/PyramidFusedNet_6/softmax/Reshape/shape:0
.while/cond/PyramidFusedNet_6/softmax/Reshape:0
0while/cond/PyramidFusedNet_6/softmax/Reshape_1:0
,while/cond/PyramidFusedNet_6/softmax/Shape:0
.while/cond/PyramidFusedNet_6/softmax/Softmax:0
2while/cond/PyramidFusedNet_6/strided_slice/stack:0
4while/cond/PyramidFusedNet_6/strided_slice/stack_1:0
4while/cond/PyramidFusedNet_6/strided_slice/stack_2:0
,while/cond/PyramidFusedNet_6/strided_slice:0
while/cond/Switch:0
while/cond/Switch:1
$while/cond/batch_decode/Decode/Exp:0
&while/cond/batch_decode/Decode/Exp_1:0
$while/cond/batch_decode/Decode/add:0
&while/cond/batch_decode/Decode/add_1:0
&while/cond/batch_decode/Decode/add_2:0
&while/cond/batch_decode/Decode/add_3:0
Ewhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/add:0
Gwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/add_1:0
Ewhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/sub:0
Gwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/sub_1:0
Wwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range/delta:0
Wwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range/start:0
Qwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range:0
Pwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Rank:0
Qwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/sub/y:0
Owhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/sub:0
Qwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/sub_1:0
Kwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose:0
Kwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv/y:0
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv:0
Mwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv_1/y:0
Kwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv_1:0
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:0
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:1
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:2
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:3
$while/cond/batch_decode/Decode/mul:0
&while/cond/batch_decode/Decode/mul_1:0
&while/cond/batch_decode/Decode/mul_2:0
&while/cond/batch_decode/Decode/mul_3:0
&while/cond/batch_decode/Decode/stack:0
$while/cond/batch_decode/Decode/sub:0
&while/cond/batch_decode/Decode/sub_1:0
6while/cond/batch_decode/Decode/transpose/Range/delta:0
6while/cond/batch_decode/Decode/transpose/Range/start:0
0while/cond/batch_decode/Decode/transpose/Range:0
/while/cond/batch_decode/Decode/transpose/Rank:0
0while/cond/batch_decode/Decode/transpose/sub/y:0
.while/cond/batch_decode/Decode/transpose/sub:0
0while/cond/batch_decode/Decode/transpose/sub_1:0
*while/cond/batch_decode/Decode/transpose:0
8while/cond/batch_decode/Decode/transpose_1/Range/delta:0
8while/cond/batch_decode/Decode/transpose_1/Range/start:0
2while/cond/batch_decode/Decode/transpose_1/Range:0
1while/cond/batch_decode/Decode/transpose_1/Rank:0
2while/cond/batch_decode/Decode/transpose_1/sub/y:0
0while/cond/batch_decode/Decode/transpose_1/sub:0
2while/cond/batch_decode/Decode/transpose_1/sub_1:0
,while/cond/batch_decode/Decode/transpose_1:0
*while/cond/batch_decode/Decode/truediv/y:0
(while/cond/batch_decode/Decode/truediv:0
,while/cond/batch_decode/Decode/truediv_1/y:0
*while/cond/batch_decode/Decode/truediv_1:0
,while/cond/batch_decode/Decode/truediv_2/y:0
*while/cond/batch_decode/Decode/truediv_2:0
,while/cond/batch_decode/Decode/truediv_3/y:0
*while/cond/batch_decode/Decode/truediv_3:0
(while/cond/batch_decode/Decode/unstack:0
(while/cond/batch_decode/Decode/unstack:1
(while/cond/batch_decode/Decode/unstack:2
(while/cond/batch_decode/Decode/unstack:3
while/cond/batch_decode/Rank:0
 while/cond/batch_decode/Rank_1:0
'while/cond/batch_decode/Reshape/shape:0
!while/cond/batch_decode/Reshape:0
)while/cond/batch_decode/Reshape_1/shape:0
#while/cond/batch_decode/Reshape_1:0
*while/cond/batch_decode/assert_equal/All:0
;while/cond/batch_decode/assert_equal/Assert/Assert/data_0:0
;while/cond/batch_decode/assert_equal/Assert/Assert/data_1:0
;while/cond/batch_decode/assert_equal/Assert/Assert/data_2:0
;while/cond/batch_decode/assert_equal/Assert/Assert/data_4:0
3while/cond/batch_decode/assert_equal/Assert/Const:0
5while/cond/batch_decode/assert_equal/Assert/Const_1:0
5while/cond/batch_decode/assert_equal/Assert/Const_2:0
5while/cond/batch_decode/assert_equal/Assert/Const_3:0
,while/cond/batch_decode/assert_equal/Const:0
,while/cond/batch_decode/assert_equal/Equal:0
(while/cond/batch_decode/assert_equal/y:0
,while/cond/batch_decode/assert_equal_1/All:0
=while/cond/batch_decode/assert_equal_1/Assert/Assert/data_0:0
=while/cond/batch_decode/assert_equal_1/Assert/Assert/data_1:0
=while/cond/batch_decode/assert_equal_1/Assert/Assert/data_2:0
=while/cond/batch_decode/assert_equal_1/Assert/Assert/data_4:0
5while/cond/batch_decode/assert_equal_1/Assert/Const:0
7while/cond/batch_decode/assert_equal_1/Assert/Const_1:0
7while/cond/batch_decode/assert_equal_1/Assert/Const_2:0
7while/cond/batch_decode/assert_equal_1/Assert/Const_3:0
.while/cond/batch_decode/assert_equal_1/Const:0
.while/cond/batch_decode/assert_equal_1/Equal:0
*while/cond/batch_decode/assert_equal_1/y:0
1while/cond/batch_decode/nms_batch/GatherV2/axis:0
,while/cond/batch_decode/nms_batch/GatherV2:0
3while/cond/batch_decode/nms_batch/GatherV2_1/axis:0
.while/cond/batch_decode/nms_batch/GatherV2_1:0
3while/cond/batch_decode/nms_batch/batch_pad/Const:0
1while/cond/batch_decode/nms_batch/batch_pad/Max:0
3while/cond/batch_decode/nms_batch/batch_pad/Shape:0
@while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Maximum/y:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Maximum:0
:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Pad:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Reshape:0
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Shape:0
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack:0
@while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_1/1:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_1:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_2:0
@while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_3/1:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_3:0
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub/y:0
:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub:0
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub_1:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/unstack:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/unstack:1
3while/cond/batch_decode/nms_batch/batch_pad/stack:0
5while/cond/batch_decode/nms_batch/batch_pad/stack_1:0
Awhile/cond/batch_decode/nms_batch/batch_pad/strided_slice/stack:0
Cwhile/cond/batch_decode/nms_batch/batch_pad/strided_slice/stack_1:0
Cwhile/cond/batch_decode/nms_batch/batch_pad/strided_slice/stack_2:0
;while/cond/batch_decode/nms_batch/batch_pad/strided_slice:0
5while/cond/batch_decode/nms_batch/batch_pad_1/Const:0
3while/cond/batch_decode/nms_batch/batch_pad_1/Max:0
5while/cond/batch_decode/nms_batch/batch_pad_1/Shape:0
Bwhile/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Maximum/y:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Maximum:0
<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Pad:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Reshape:0
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Shape:0
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_1:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_2:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_3:0
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub/y:0
<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub:0
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub_1:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/unstack:0
5while/cond/batch_decode/nms_batch/batch_pad_1/stack:0
7while/cond/batch_decode/nms_batch/batch_pad_1/stack_1:0
Cwhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stack:0
Ewhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stack_1:0
Ewhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stack_2:0
=while/cond/batch_decode/nms_batch/batch_pad_1/strided_slice:0
5while/cond/batch_decode/nms_batch/nms/GatherV2/axis:0
0while/cond/batch_decode/nms_batch/nms/GatherV2:0
/while/cond/batch_decode/nms_batch/nms/Greater:0
Kwhile/cond/batch_decode/nms_batch/nms/NonMaxSuppressionV2/max_output_size:0
;while/cond/batch_decode/nms_batch/nms/NonMaxSuppressionV2:0
5while/cond/batch_decode/nms_batch/nms/Reshape/shape:0
/while/cond/batch_decode/nms_batch/nms/Reshape:0
-while/cond/batch_decode/nms_batch/nms/Where:0
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/GatherV2/axis:0
=while/cond/batch_decode/nms_batch/nms/boolean_mask/GatherV2:0
Kwhile/cond/batch_decode/nms_batch/nms/boolean_mask/Prod/reduction_indices:0
9while/cond/batch_decode/nms_batch/nms/boolean_mask/Prod:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape_1/shape:0
>while/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape_1:0
:while/cond/batch_decode/nms_batch/nms/boolean_mask/Shape:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask/Shape_1:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask/Shape_2:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask/Squeeze:0
:while/cond/batch_decode/nms_batch/nms/boolean_mask/Where:0
@while/cond/batch_decode/nms_batch/nms/boolean_mask/concat/axis:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask/concat/values_1:0
;while/cond/batch_decode/nms_batch/nms/boolean_mask/concat:0
Hwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stack:0
Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stack_1:0
Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stack_2:0
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice:0
Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stack:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stack_1:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stack_2:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1:0
Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stack:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stack_1:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stack_2:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/GatherV2/axis:0
?while/cond/batch_decode/nms_batch/nms/boolean_mask_1/GatherV2:0
Mwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/Prod/reduction_indices:0
;while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Prod:0
>while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape:0
Fwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape_1/shape:0
@while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape_1:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Shape:0
>while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Shape_1:0
>while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Shape_2:0
>while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Squeeze:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Where:0
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat/axis:0
Fwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat/values_1:0
=while/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat:0
Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stack:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stack_1:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stack_2:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stack:0
Nwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1:0
Nwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2:0
Fwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stack:0
Nwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1:0
Nwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2:0
Fwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2:0
5while/cond/batch_decode/nms_batch/nms/iou_threshold:0
7while/cond/batch_decode/nms_batch/nms/score_threshold:0
7while/cond/batch_decode/nms_batch/strided_slice/stack:0
9while/cond/batch_decode/nms_batch/strided_slice/stack_1:0
9while/cond/batch_decode/nms_batch/strided_slice/stack_2:0
1while/cond/batch_decode/nms_batch/strided_slice:0
9while/cond/batch_decode/nms_batch/strided_slice_1/stack:0
;while/cond/batch_decode/nms_batch/strided_slice_1/stack_1:0
;while/cond/batch_decode/nms_batch/strided_slice_1/stack_2:0
3while/cond/batch_decode/nms_batch/strided_slice_1:0
9while/cond/batch_decode/nms_batch/strided_slice_2/stack:0
;while/cond/batch_decode/nms_batch/strided_slice_2/stack_1:0
;while/cond/batch_decode/nms_batch/strided_slice_2/stack_2:0
3while/cond/batch_decode/nms_batch/strided_slice_2:0
9while/cond/batch_decode/nms_batch/strided_slice_3/stack:0
;while/cond/batch_decode/nms_batch/strided_slice_3/stack_1:0
;while/cond/batch_decode/nms_batch/strided_slice_3/stack_2:0
3while/cond/batch_decode/nms_batch/strided_slice_3:0
while/cond/batch_decode/stack:0
!while/cond/batch_decode/unstack:0
&while/cond/batch_decode_1/Decode/Exp:0
(while/cond/batch_decode_1/Decode/Exp_1:0
&while/cond/batch_decode_1/Decode/add:0
(while/cond/batch_decode_1/Decode/add_1:0
(while/cond/batch_decode_1/Decode/add_2:0
(while/cond/batch_decode_1/Decode/add_3:0
Gwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/add:0
Iwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/add_1:0
Gwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/sub:0
Iwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/sub_1:0
Ywhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range/delta:0
Ywhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range/start:0
Swhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range:0
Rwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Rank:0
Swhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/sub/y:0
Qwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/sub:0
Swhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/sub_1:0
Mwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose:0
Mwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv/y:0
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv:0
Owhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv_1/y:0
Mwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv_1:0
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:0
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:1
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:2
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:3
&while/cond/batch_decode_1/Decode/mul:0
(while/cond/batch_decode_1/Decode/mul_1:0
(while/cond/batch_decode_1/Decode/mul_2:0
(while/cond/batch_decode_1/Decode/mul_3:0
(while/cond/batch_decode_1/Decode/stack:0
&while/cond/batch_decode_1/Decode/sub:0
(while/cond/batch_decode_1/Decode/sub_1:0
8while/cond/batch_decode_1/Decode/transpose/Range/delta:0
8while/cond/batch_decode_1/Decode/transpose/Range/start:0
2while/cond/batch_decode_1/Decode/transpose/Range:0
1while/cond/batch_decode_1/Decode/transpose/Rank:0
2while/cond/batch_decode_1/Decode/transpose/sub/y:0
0while/cond/batch_decode_1/Decode/transpose/sub:0
2while/cond/batch_decode_1/Decode/transpose/sub_1:0
,while/cond/batch_decode_1/Decode/transpose:0
:while/cond/batch_decode_1/Decode/transpose_1/Range/delta:0
:while/cond/batch_decode_1/Decode/transpose_1/Range/start:0
4while/cond/batch_decode_1/Decode/transpose_1/Range:0
3while/cond/batch_decode_1/Decode/transpose_1/Rank:0
4while/cond/batch_decode_1/Decode/transpose_1/sub/y:0
2while/cond/batch_decode_1/Decode/transpose_1/sub:0
4while/cond/batch_decode_1/Decode/transpose_1/sub_1:0
.while/cond/batch_decode_1/Decode/transpose_1:0
,while/cond/batch_decode_1/Decode/truediv/y:0
*while/cond/batch_decode_1/Decode/truediv:0
.while/cond/batch_decode_1/Decode/truediv_1/y:0
,while/cond/batch_decode_1/Decode/truediv_1:0
.while/cond/batch_decode_1/Decode/truediv_2/y:0
,while/cond/batch_decode_1/Decode/truediv_2:0
.while/cond/batch_decode_1/Decode/truediv_3/y:0
,while/cond/batch_decode_1/Decode/truediv_3:0
*while/cond/batch_decode_1/Decode/unstack:0
*while/cond/batch_decode_1/Decode/unstack:1
*while/cond/batch_decode_1/Decode/unstack:2
*while/cond/batch_decode_1/Decode/unstack:3
 while/cond/batch_decode_1/Rank:0
"while/cond/batch_decode_1/Rank_1:0
)while/cond/batch_decode_1/Reshape/shape:0
#while/cond/batch_decode_1/Reshape:0
+while/cond/batch_decode_1/Reshape_1/shape:0
%while/cond/batch_decode_1/Reshape_1:0
,while/cond/batch_decode_1/assert_equal/All:0
=while/cond/batch_decode_1/assert_equal/Assert/Assert/data_0:0
=while/cond/batch_decode_1/assert_equal/Assert/Assert/data_1:0
=while/cond/batch_decode_1/assert_equal/Assert/Assert/data_2:0
=while/cond/batch_decode_1/assert_equal/Assert/Assert/data_4:0
5while/cond/batch_decode_1/assert_equal/Assert/Const:0
7while/cond/batch_decode_1/assert_equal/Assert/Const_1:0
7while/cond/batch_decode_1/assert_equal/Assert/Const_2:0
7while/cond/batch_decode_1/assert_equal/Assert/Const_3:0
.while/cond/batch_decode_1/assert_equal/Const:0
.while/cond/batch_decode_1/assert_equal/Equal:0
*while/cond/batch_decode_1/assert_equal/y:0
.while/cond/batch_decode_1/assert_equal_1/All:0
?while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_0:0
?while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_1:0
?while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_2:0
?while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_4:0
7while/cond/batch_decode_1/assert_equal_1/Assert/Const:0
9while/cond/batch_decode_1/assert_equal_1/Assert/Const_1:0
9while/cond/batch_decode_1/assert_equal_1/Assert/Const_2:0
9while/cond/batch_decode_1/assert_equal_1/Assert/Const_3:0
0while/cond/batch_decode_1/assert_equal_1/Const:0
0while/cond/batch_decode_1/assert_equal_1/Equal:0
,while/cond/batch_decode_1/assert_equal_1/y:0
3while/cond/batch_decode_1/nms_batch/GatherV2/axis:0
.while/cond/batch_decode_1/nms_batch/GatherV2:0
5while/cond/batch_decode_1/nms_batch/GatherV2_1/axis:0
0while/cond/batch_decode_1/nms_batch/GatherV2_1:0
5while/cond/batch_decode_1/nms_batch/batch_pad/Const:0
3while/cond/batch_decode_1/nms_batch/batch_pad/Max:0
5while/cond/batch_decode_1/nms_batch/batch_pad/Shape:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Maximum/y:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Maximum:0
<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Pad:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Reshape:0
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Shape:0
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_1/1:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_1:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_2:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_3/1:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_3:0
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub/y:0
<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub:0
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub_1:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/unstack:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/unstack:1
5while/cond/batch_decode_1/nms_batch/batch_pad/stack:0
7while/cond/batch_decode_1/nms_batch/batch_pad/stack_1:0
Cwhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stack:0
Ewhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stack_1:0
Ewhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stack_2:0
=while/cond/batch_decode_1/nms_batch/batch_pad/strided_slice:0
7while/cond/batch_decode_1/nms_batch/batch_pad_1/Const:0
5while/cond/batch_decode_1/nms_batch/batch_pad_1/Max:0
7while/cond/batch_decode_1/nms_batch/batch_pad_1/Shape:0
Dwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Maximum/y:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Maximum:0
>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Pad:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Reshape:0
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Shape:0
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_1:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_2:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_3:0
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub/y:0
>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub:0
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub_1:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/unstack:0
7while/cond/batch_decode_1/nms_batch/batch_pad_1/stack:0
9while/cond/batch_decode_1/nms_batch/batch_pad_1/stack_1:0
Ewhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stack:0
Gwhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stack_1:0
Gwhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stack_2:0
?while/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice:0
7while/cond/batch_decode_1/nms_batch/nms/GatherV2/axis:0
2while/cond/batch_decode_1/nms_batch/nms/GatherV2:0
1while/cond/batch_decode_1/nms_batch/nms/Greater:0
Mwhile/cond/batch_decode_1/nms_batch/nms/NonMaxSuppressionV2/max_output_size:0
=while/cond/batch_decode_1/nms_batch/nms/NonMaxSuppressionV2:0
7while/cond/batch_decode_1/nms_batch/nms/Reshape/shape:0
1while/cond/batch_decode_1/nms_batch/nms/Reshape:0
/while/cond/batch_decode_1/nms_batch/nms/Where:0
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/GatherV2/axis:0
?while/cond/batch_decode_1/nms_batch/nms/boolean_mask/GatherV2:0
Mwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/Prod/reduction_indices:0
;while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Prod:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape_1/shape:0
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape_1:0
<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Shape:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Shape_1:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Shape_2:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Squeeze:0
<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Where:0
Bwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat/axis:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat/values_1:0
=while/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat:0
Jwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stack:0
Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stack_1:0
Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stack_2:0
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice:0
Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stack:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stack_1:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stack_2:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1:0
Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stack:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stack_1:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stack_2:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/GatherV2/axis:0
Awhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/GatherV2:0
Owhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Prod/reduction_indices:0
=while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Prod:0
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape:0
Hwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape_1/shape:0
Bwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape_1:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Shape:0
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Shape_1:0
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Shape_2:0
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Squeeze:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Where:0
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat/axis:0
Hwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat/values_1:0
?while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat:0
Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stack:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stack_1:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stack_2:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stack:0
Pwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1:0
Pwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2:0
Hwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stack:0
Pwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1:0
Pwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2:0
Hwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2:0
7while/cond/batch_decode_1/nms_batch/nms/iou_threshold:0
9while/cond/batch_decode_1/nms_batch/nms/score_threshold:0
9while/cond/batch_decode_1/nms_batch/strided_slice/stack:0
;while/cond/batch_decode_1/nms_batch/strided_slice/stack_1:0
;while/cond/batch_decode_1/nms_batch/strided_slice/stack_2:0
3while/cond/batch_decode_1/nms_batch/strided_slice:0
;while/cond/batch_decode_1/nms_batch/strided_slice_1/stack:0
=while/cond/batch_decode_1/nms_batch/strided_slice_1/stack_1:0
=while/cond/batch_decode_1/nms_batch/strided_slice_1/stack_2:0
5while/cond/batch_decode_1/nms_batch/strided_slice_1:0
;while/cond/batch_decode_1/nms_batch/strided_slice_2/stack:0
=while/cond/batch_decode_1/nms_batch/strided_slice_2/stack_1:0
=while/cond/batch_decode_1/nms_batch/strided_slice_2/stack_2:0
5while/cond/batch_decode_1/nms_batch/strided_slice_2:0
;while/cond/batch_decode_1/nms_batch/strided_slice_3/stack:0
=while/cond/batch_decode_1/nms_batch/strided_slice_3/stack_1:0
=while/cond/batch_decode_1/nms_batch/strided_slice_3/stack_2:0
5while/cond/batch_decode_1/nms_batch/strided_slice_3:0
!while/cond/batch_decode_1/stack:0
#while/cond/batch_decode_1/unstack:0
&while/cond/batch_decode_2/Decode/Exp:0
(while/cond/batch_decode_2/Decode/Exp_1:0
&while/cond/batch_decode_2/Decode/add:0
(while/cond/batch_decode_2/Decode/add_1:0
(while/cond/batch_decode_2/Decode/add_2:0
(while/cond/batch_decode_2/Decode/add_3:0
Gwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/add:0
Iwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/add_1:0
Gwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/sub:0
Iwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/sub_1:0
Ywhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range/delta:0
Ywhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range/start:0
Swhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range:0
Rwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Rank:0
Swhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/sub/y:0
Qwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/sub:0
Swhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/sub_1:0
Mwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose:0
Mwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv/y:0
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv:0
Owhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv_1/y:0
Mwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv_1:0
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:0
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:1
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:2
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:3
&while/cond/batch_decode_2/Decode/mul:0
(while/cond/batch_decode_2/Decode/mul_1:0
(while/cond/batch_decode_2/Decode/mul_2:0
(while/cond/batch_decode_2/Decode/mul_3:0
(while/cond/batch_decode_2/Decode/stack:0
&while/cond/batch_decode_2/Decode/sub:0
(while/cond/batch_decode_2/Decode/sub_1:0
8while/cond/batch_decode_2/Decode/transpose/Range/delta:0
8while/cond/batch_decode_2/Decode/transpose/Range/start:0
2while/cond/batch_decode_2/Decode/transpose/Range:0
1while/cond/batch_decode_2/Decode/transpose/Rank:0
2while/cond/batch_decode_2/Decode/transpose/sub/y:0
0while/cond/batch_decode_2/Decode/transpose/sub:0
2while/cond/batch_decode_2/Decode/transpose/sub_1:0
,while/cond/batch_decode_2/Decode/transpose:0
:while/cond/batch_decode_2/Decode/transpose_1/Range/delta:0
:while/cond/batch_decode_2/Decode/transpose_1/Range/start:0
4while/cond/batch_decode_2/Decode/transpose_1/Range:0
3while/cond/batch_decode_2/Decode/transpose_1/Rank:0
4while/cond/batch_decode_2/Decode/transpose_1/sub/y:0
2while/cond/batch_decode_2/Decode/transpose_1/sub:0
4while/cond/batch_decode_2/Decode/transpose_1/sub_1:0
.while/cond/batch_decode_2/Decode/transpose_1:0
,while/cond/batch_decode_2/Decode/truediv/y:0
*while/cond/batch_decode_2/Decode/truediv:0
.while/cond/batch_decode_2/Decode/truediv_1/y:0
,while/cond/batch_decode_2/Decode/truediv_1:0
.while/cond/batch_decode_2/Decode/truediv_2/y:0
,while/cond/batch_decode_2/Decode/truediv_2:0
.while/cond/batch_decode_2/Decode/truediv_3/y:0
,while/cond/batch_decode_2/Decode/truediv_3:0
*while/cond/batch_decode_2/Decode/unstack:0
*while/cond/batch_decode_2/Decode/unstack:1
*while/cond/batch_decode_2/Decode/unstack:2
*while/cond/batch_decode_2/Decode/unstack:3
 while/cond/batch_decode_2/Rank:0
"while/cond/batch_decode_2/Rank_1:0
)while/cond/batch_decode_2/Reshape/shape:0
#while/cond/batch_decode_2/Reshape:0
+while/cond/batch_decode_2/Reshape_1/shape:0
%while/cond/batch_decode_2/Reshape_1:0
,while/cond/batch_decode_2/assert_equal/All:0
=while/cond/batch_decode_2/assert_equal/Assert/Assert/data_0:0
=while/cond/batch_decode_2/assert_equal/Assert/Assert/data_1:0
=while/cond/batch_decode_2/assert_equal/Assert/Assert/data_2:0
=while/cond/batch_decode_2/assert_equal/Assert/Assert/data_4:0
5while/cond/batch_decode_2/assert_equal/Assert/Const:0
7while/cond/batch_decode_2/assert_equal/Assert/Const_1:0
7while/cond/batch_decode_2/assert_equal/Assert/Const_2:0
7while/cond/batch_decode_2/assert_equal/Assert/Const_3:0
.while/cond/batch_decode_2/assert_equal/Const:0
.while/cond/batch_decode_2/assert_equal/Equal:0
*while/cond/batch_decode_2/assert_equal/y:0
.while/cond/batch_decode_2/assert_equal_1/All:0
?while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_0:0
?while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_1:0
?while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_2:0
?while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_4:0
7while/cond/batch_decode_2/assert_equal_1/Assert/Const:0
9while/cond/batch_decode_2/assert_equal_1/Assert/Const_1:0
9while/cond/batch_decode_2/assert_equal_1/Assert/Const_2:0
9while/cond/batch_decode_2/assert_equal_1/Assert/Const_3:0
0while/cond/batch_decode_2/assert_equal_1/Const:0
0while/cond/batch_decode_2/assert_equal_1/Equal:0
,while/cond/batch_decode_2/assert_equal_1/y:0
3while/cond/batch_decode_2/nms_batch/GatherV2/axis:0
.while/cond/batch_decode_2/nms_batch/GatherV2:0
5while/cond/batch_decode_2/nms_batch/GatherV2_1/axis:0
0while/cond/batch_decode_2/nms_batch/GatherV2_1:0
5while/cond/batch_decode_2/nms_batch/batch_pad/Const:0
3while/cond/batch_decode_2/nms_batch/batch_pad/Max:0
5while/cond/batch_decode_2/nms_batch/batch_pad/Shape:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Maximum/y:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Maximum:0
<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Pad:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Reshape:0
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Shape:0
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_1/1:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_1:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_2:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_3/1:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_3:0
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub/y:0
<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub:0
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub_1:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/unstack:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/unstack:1
5while/cond/batch_decode_2/nms_batch/batch_pad/stack:0
7while/cond/batch_decode_2/nms_batch/batch_pad/stack_1:0
Cwhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stack:0
Ewhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stack_1:0
Ewhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stack_2:0
=while/cond/batch_decode_2/nms_batch/batch_pad/strided_slice:0
7while/cond/batch_decode_2/nms_batch/batch_pad_1/Const:0
5while/cond/batch_decode_2/nms_batch/batch_pad_1/Max:0
7while/cond/batch_decode_2/nms_batch/batch_pad_1/Shape:0
Dwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Maximum/y:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Maximum:0
>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Pad:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Reshape:0
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Shape:0
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_1:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_2:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_3:0
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub/y:0
>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub:0
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub_1:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/unstack:0
7while/cond/batch_decode_2/nms_batch/batch_pad_1/stack:0
9while/cond/batch_decode_2/nms_batch/batch_pad_1/stack_1:0
Ewhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stack:0
Gwhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stack_1:0
Gwhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stack_2:0
?while/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice:0
7while/cond/batch_decode_2/nms_batch/nms/GatherV2/axis:0
2while/cond/batch_decode_2/nms_batch/nms/GatherV2:0
1while/cond/batch_decode_2/nms_batch/nms/Greater:0
Mwhile/cond/batch_decode_2/nms_batch/nms/NonMaxSuppressionV2/max_output_size:0
=while/cond/batch_decode_2/nms_batch/nms/NonMaxSuppressionV2:0
7while/cond/batch_decode_2/nms_batch/nms/Reshape/shape:0
1while/cond/batch_decode_2/nms_batch/nms/Reshape:0
/while/cond/batch_decode_2/nms_batch/nms/Where:0
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/GatherV2/axis:0
?while/cond/batch_decode_2/nms_batch/nms/boolean_mask/GatherV2:0
Mwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/Prod/reduction_indices:0
;while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Prod:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape_1/shape:0
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape_1:0
<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Shape:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Shape_1:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Shape_2:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Squeeze:0
<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Where:0
Bwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat/axis:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat/values_1:0
=while/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat:0
Jwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stack:0
Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stack_1:0
Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stack_2:0
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice:0
Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stack:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stack_1:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stack_2:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1:0
Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stack:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stack_1:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stack_2:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/GatherV2/axis:0
Awhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/GatherV2:0
Owhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Prod/reduction_indices:0
=while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Prod:0
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape:0
Hwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape_1/shape:0
Bwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape_1:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Shape:0
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Shape_1:0
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Shape_2:0
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Squeeze:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Where:0
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat/axis:0
Hwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat/values_1:0
?while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat:0
Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stack:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stack_1:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stack_2:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stack:0
Pwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1:0
Pwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2:0
Hwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stack:0
Pwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1:0
Pwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2:0
Hwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2:0
7while/cond/batch_decode_2/nms_batch/nms/iou_threshold:0
9while/cond/batch_decode_2/nms_batch/nms/score_threshold:0
9while/cond/batch_decode_2/nms_batch/strided_slice/stack:0
;while/cond/batch_decode_2/nms_batch/strided_slice/stack_1:0
;while/cond/batch_decode_2/nms_batch/strided_slice/stack_2:0
3while/cond/batch_decode_2/nms_batch/strided_slice:0
;while/cond/batch_decode_2/nms_batch/strided_slice_1/stack:0
=while/cond/batch_decode_2/nms_batch/strided_slice_1/stack_1:0
=while/cond/batch_decode_2/nms_batch/strided_slice_1/stack_2:0
5while/cond/batch_decode_2/nms_batch/strided_slice_1:0
;while/cond/batch_decode_2/nms_batch/strided_slice_2/stack:0
=while/cond/batch_decode_2/nms_batch/strided_slice_2/stack_1:0
=while/cond/batch_decode_2/nms_batch/strided_slice_2/stack_2:0
5while/cond/batch_decode_2/nms_batch/strided_slice_2:0
;while/cond/batch_decode_2/nms_batch/strided_slice_3/stack:0
=while/cond/batch_decode_2/nms_batch/strided_slice_3/stack_1:0
=while/cond/batch_decode_2/nms_batch/strided_slice_3/stack_2:0
5while/cond/batch_decode_2/nms_batch/strided_slice_3:0
!while/cond/batch_decode_2/stack:0
#while/cond/batch_decode_2/unstack:0
while/cond/concat/axis:0
while/cond/concat:0
while/cond/concat_1/axis:0
while/cond/concat_1:0
while/cond/concat_2/axis:0
while/cond/concat_2:0
while/cond/concat_3/axis:0
while/cond/concat_3:0
while/cond/pred_id:0
while/cond/switch_f:0
while/cond/switch_t:0
while/img_shape/Equal/y:0
while/img_shape/Equal:0
while/img_shape/Rank:0
&while/img_shape/assert_rank_in/Shape:0
%while/img_shape/assert_rank_in/rank:0
'while/img_shape/assert_rank_in/rank_1:0
while/img_shape/cond/Merge:0
while/img_shape/cond/Merge:1
while/img_shape/cond/Shape:0
while/img_shape/cond/Shape_1:0
while/img_shape/cond/Switch:0
while/img_shape/cond/Switch:1
while/img_shape/cond/pred_id:0
*while/img_shape/cond/strided_slice/stack:0
,while/img_shape/cond/strided_slice/stack_1:0
,while/img_shape/cond/strided_slice/stack_2:0
$while/img_shape/cond/strided_slice:0
,while/img_shape/cond/strided_slice_1/stack:0
.while/img_shape/cond/strided_slice_1/stack_1:0
.while/img_shape/cond/strided_slice_1/stack_2:0
&while/img_shape/cond/strided_slice_1:0
while/img_shape/cond/switch_f:0
while/img_shape/cond/switch_t:0
while/img_shape_1/Equal/y:0
while/img_shape_1/Equal:0
while/img_shape_1/Rank:0
(while/img_shape_1/assert_rank_in/Shape:0
'while/img_shape_1/assert_rank_in/rank:0
)while/img_shape_1/assert_rank_in/rank_1:0
while/img_shape_1/cond/Merge:0
while/img_shape_1/cond/Merge:1
while/img_shape_1/cond/Shape:0
 while/img_shape_1/cond/Shape_1:0
while/img_shape_1/cond/Switch:0
while/img_shape_1/cond/Switch:1
 while/img_shape_1/cond/pred_id:0
,while/img_shape_1/cond/strided_slice/stack:0
.while/img_shape_1/cond/strided_slice/stack_1:0
.while/img_shape_1/cond/strided_slice/stack_2:0
&while/img_shape_1/cond/strided_slice:0
.while/img_shape_1/cond/strided_slice_1/stack:0
0while/img_shape_1/cond/strided_slice_1/stack_1:0
0while/img_shape_1/cond/strided_slice_1/stack_2:0
(while/img_shape_1/cond/strided_slice_1:0
!while/img_shape_1/cond/switch_f:0
!while/img_shape_1/cond/switch_t:0
while/img_shape_2/Equal/y:0
while/img_shape_2/Equal:0
while/img_shape_2/Rank:0
(while/img_shape_2/assert_rank_in/Shape:0
'while/img_shape_2/assert_rank_in/rank:0
)while/img_shape_2/assert_rank_in/rank_1:0
while/img_shape_2/cond/Merge:0
while/img_shape_2/cond/Merge:1
while/img_shape_2/cond/Shape:0
 while/img_shape_2/cond/Shape_1:0
while/img_shape_2/cond/Switch:0
while/img_shape_2/cond/Switch:1
 while/img_shape_2/cond/pred_id:0
,while/img_shape_2/cond/strided_slice/stack:0
.while/img_shape_2/cond/strided_slice/stack_1:0
.while/img_shape_2/cond/strided_slice/stack_2:0
&while/img_shape_2/cond/strided_slice:0
.while/img_shape_2/cond/strided_slice_1/stack:0
0while/img_shape_2/cond/strided_slice_1/stack_1:0
0while/img_shape_2/cond/strided_slice_1/stack_2:0
(while/img_shape_2/cond/strided_slice_1:0
!while/img_shape_2/cond/switch_f:0
!while/img_shape_2/cond/switch_t:0
while/img_shape_3/Equal/y:0
while/img_shape_3/Equal:0
while/img_shape_3/Rank:0
(while/img_shape_3/assert_rank_in/Shape:0
'while/img_shape_3/assert_rank_in/rank:0
)while/img_shape_3/assert_rank_in/rank_1:0
while/img_shape_3/cond/Merge:0
while/img_shape_3/cond/Merge:1
while/img_shape_3/cond/Shape:0
 while/img_shape_3/cond/Shape_1:0
while/img_shape_3/cond/Switch:0
while/img_shape_3/cond/Switch:1
 while/img_shape_3/cond/pred_id:0
,while/img_shape_3/cond/strided_slice/stack:0
.while/img_shape_3/cond/strided_slice/stack_1:0
.while/img_shape_3/cond/strided_slice/stack_2:0
&while/img_shape_3/cond/strided_slice:0
.while/img_shape_3/cond/strided_slice_1/stack:0
0while/img_shape_3/cond/strided_slice_1/stack_1:0
0while/img_shape_3/cond/strided_slice_1/stack_2:0
(while/img_shape_3/cond/strided_slice_1:0
!while/img_shape_3/cond/switch_f:0
!while/img_shape_3/cond/switch_t:0
while/meshgrid/Reshape/shape:0
while/meshgrid/Reshape:0
 while/meshgrid/Reshape_1/shape:0
while/meshgrid/Reshape_1:0
 while/meshgrid/Reshape_2/shape:0
while/meshgrid/Reshape_2:0
 while/meshgrid/Reshape_3/shape:0
while/meshgrid/Reshape_3:0
while/meshgrid/Size:0
while/meshgrid/Size_1:0
while/meshgrid/mul:0
while/meshgrid/mul_1:0
while/meshgrid/ones/Const:0
while/meshgrid/ones/Less/y:0
while/meshgrid/ones/Less:0
while/meshgrid/ones/mul:0
while/meshgrid/ones/packed:0
while/meshgrid/ones:0
 while/meshgrid_1/Reshape/shape:0
while/meshgrid_1/Reshape:0
"while/meshgrid_1/Reshape_1/shape:0
while/meshgrid_1/Reshape_1:0
"while/meshgrid_1/Reshape_2/shape:0
while/meshgrid_1/Reshape_2:0
"while/meshgrid_1/Reshape_3/shape:0
while/meshgrid_1/Reshape_3:0
while/meshgrid_1/Size:0
while/meshgrid_1/Size_1:0
while/meshgrid_1/mul:0
while/meshgrid_1/mul_1:0
while/meshgrid_1/ones/Const:0
while/meshgrid_1/ones/Less/y:0
while/meshgrid_1/ones/Less:0
while/meshgrid_1/ones/mul:0
while/meshgrid_1/ones/packed:0
while/meshgrid_1/ones:0
while/mul/y:0
while/mul:0
while/mul_1/y:0
while/mul_1:0
while/mul_2/y:0
while/mul_2:0
while/mul_3/y:0
while/mul_3:0
while/range/Cast:0
while/range/Cast_1:0
while/range/delta:0
while/range/start:0
while/range:0
while/range_1/Cast:0
while/range_1/Cast_1:0
while/range_1/delta:0
while/range_1/start:0
while/range_1:0
while/range_2/Cast:0
while/range_2/Cast_1:0
while/range_2/delta:0
while/range_2/start:0
while/range_2:0
while/range_3/Cast:0
while/range_3/Cast_1:0
while/range_3/delta:0
while/range_3/start:0
while/range_3:0
while/stack:0
while/stack_1:0
while/stack_2:0
while/stack_3:0
while/stack_4:0
while/strided_slice/stack:0
while/strided_slice/stack_1:0
while/strided_slice/stack_2:0
while/strided_slice:0
while/strided_slice_1/stack:0
while/strided_slice_1/stack_1:0
while/strided_slice_1/stack_2:0
while/strided_slice_10/stack:0
 while/strided_slice_10/stack_1:0
 while/strided_slice_10/stack_2:0
while/strided_slice_10:0
while/strided_slice_11/stack:0
 while/strided_slice_11/stack_1:0
 while/strided_slice_11/stack_2:0
while/strided_slice_11:0
while/strided_slice_12/stack:0
 while/strided_slice_12/stack_1:0
 while/strided_slice_12/stack_2:0
while/strided_slice_12:0
while/strided_slice_13/stack:0
 while/strided_slice_13/stack_1:0
 while/strided_slice_13/stack_2:0
while/strided_slice_13:0
while/strided_slice_14/stack:0
 while/strided_slice_14/stack_1:0
 while/strided_slice_14/stack_2:0
while/strided_slice_14:0
while/strided_slice_15/stack:0
 while/strided_slice_15/stack_1:0
 while/strided_slice_15/stack_2:0
while/strided_slice_15:0
while/strided_slice_1:0
while/strided_slice_2/stack:0
while/strided_slice_2/stack_1:0
while/strided_slice_2/stack_2:0
while/strided_slice_2:0
while/strided_slice_3/stack:0
while/strided_slice_3/stack_1:0
while/strided_slice_3/stack_2:0
while/strided_slice_3:0
while/strided_slice_4/stack:0
while/strided_slice_4/stack_1:0
while/strided_slice_4/stack_2:0
while/strided_slice_4:0
while/strided_slice_5/stack:0
while/strided_slice_5/stack_1:0
while/strided_slice_5/stack_2:0
while/strided_slice_5:0
while/strided_slice_6/stack:0
while/strided_slice_6/stack_1:0
while/strided_slice_6/stack_2:0
while/strided_slice_6:0
while/strided_slice_7/stack:0
while/strided_slice_7/stack_1:0
while/strided_slice_7/stack_2:0
while/strided_slice_7:0
while/strided_slice_8/stack:0
while/strided_slice_8/stack_1:0
while/strided_slice_8/stack_2:0
while/strided_slice_8:0
while/strided_slice_9/stack:0
while/strided_slice_9/stack_1:0
while/strided_slice_9/stack_2:0
while/strided_slice_9:0
while/sub/y:0
while/sub:0
while/sub_1/y:0
while/sub_1:0
while/sub_2/y:0
while/sub_2:0
while/sub_3/y:0
while/sub_3:0
while/truediv/Cast:0
while/truediv/Cast_1:0
while/truediv/y:0
while/truediv:0
while/truediv_1/Cast:0
while/truediv_1/Cast_1:0
while/truediv_1/y:0
while/truediv_1:0
while/truediv_2/Cast:0
while/truediv_2/Cast_1:0
while/truediv_2/y:0
while/truediv_2:0
while/truediv_3/Cast:0
while/truediv_3/Cast_1:0
while/truediv_3/y:0
while/truediv_3:0
while/truediv_4/Cast:0
while/truediv_4/Cast_1:0
while/truediv_4/y:0
while/truediv_4:0
while/truediv_5:0
while/truediv_6:0
while/truediv_7:0
while/truediv_8:0
5PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/read:0Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0
:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/read:0Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3:0
7PyramidFusedNet/dem1_log_conv0/pointwise_weights/read:0Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enter:0
;PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/read:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
7PyramidFusedNet/dem0_log_conv0/pointwise_weights/read:0Dwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Enter:0}
2PyramidFusedNet/fem_conv0/depthwise_weights/read:0Gwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enter:0
5PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/read:0Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0
:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/read:0Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3:0
;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/read:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0y
0PyramidFusedNet/fem_conv2/BatchNorm/gamma/read:0Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter:0
7PyramidFusedNet/dem1_log_conv0/depthwise_weights/read:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enter:0
?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/read:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
7PyramidFusedNet/dem0_log_conv0/depthwise_weights/read:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/Enter:0z
/PyramidFusedNet/fem_conv2/BatchNorm/beta/read:0Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1:0
?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/read:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0s
2PyramidFusedNet/fem_conv4/pointwise_weights/read:0=while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enter:0
7PyramidFusedNet/dem1_reg_conv0/pointwise_weights/read:0Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enter:0
4PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/read:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
6PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/read:0Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2:0
6PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/read:0Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
4PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/read:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
7PyramidFusedNet/dem0_reg_conv0/pointwise_weights/read:0Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Enter:0}
2PyramidFusedNet/fem_conv4/depthwise_weights/read:0Gwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enter:0
?PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/read:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0y
0PyramidFusedNet/fem_conv1/BatchNorm/gamma/read:0Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter:0
7PyramidFusedNet/dem1_reg_conv0/depthwise_weights/read:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enter:0
?PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/read:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0z
/PyramidFusedNet/fem_conv4/BatchNorm/beta/read:0Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1:0
7PyramidFusedNet/dem0_reg_conv0/depthwise_weights/read:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/Enter:0s
2PyramidFusedNet/fem_conv1/pointwise_weights/read:0=while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enter:0
7PyramidFusedNet/dem1_log_conv1/pointwise_weights/read:0Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enter:0
7PyramidFusedNet/dem0_log_conv1/pointwise_weights/read:0Dwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Enter:0}
2PyramidFusedNet/fem_conv1/depthwise_weights/read:0Gwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enter:0
?PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/read:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
7PyramidFusedNet/dem1_log_conv1/depthwise_weights/read:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enter:0
;PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/read:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0y
0PyramidFusedNet/fem_conv0/BatchNorm/gamma/read:0Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter:0
;PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/read:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/read:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0A
image_pyramid/while/Exit_1:0!while/TensorArraySizeV3/Enter_1:0
7PyramidFusedNet/dem0_log_conv1/depthwise_weights/read:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/Enter:0
?PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/read:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
5PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/read:0Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter:0
4PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/read:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
7PyramidFusedNet/dem1_reg_conv1/pointwise_weights/read:0Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enter:0
4PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/read:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
7PyramidFusedNet/dem0_reg_conv1/pointwise_weights/read:0Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Enter:0
5PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/read:0Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter:0
6PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/read:0Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2:0
7PyramidFusedNet/dem1_reg_conv1/depthwise_weights/read:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enter:0
6PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/read:0Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2:0z
/PyramidFusedNet/fem_conv1/BatchNorm/beta/read:0Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
7PyramidFusedNet/dem0_reg_conv1/depthwise_weights/read:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/Enter:0s
2PyramidFusedNet/fem_conv2/pointwise_weights/read:0=while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enter:0>
image_pyramid/TensorArray:0while/TensorArraySizeV3/Enter:0
5PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/read:0Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter:0
4PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/read:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
4PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/read:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0}
2PyramidFusedNet/fem_conv2/depthwise_weights/read:0Gwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enter:0
5PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/read:0Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter:0z
/PyramidFusedNet/fem_conv3/BatchNorm/beta/read:0Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1:0
;PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/read:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/read:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/read:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0y
0PyramidFusedNet/fem_conv4/BatchNorm/gamma/read:0Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter:0s
2PyramidFusedNet/fem_conv3/pointwise_weights/read:0=while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enter:0
6PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/read:0Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2:0
5PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/read:0Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0
4PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/read:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
5PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/read:0Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0
4PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/read:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/read:0Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3:0}
2PyramidFusedNet/fem_conv3/depthwise_weights/read:0Gwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enter:0y
0PyramidFusedNet/fem_conv3/BatchNorm/gamma/read:0Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter:0
?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/read:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/read:0Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3:0z
/PyramidFusedNet/fem_conv0/BatchNorm/beta/read:0Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/read:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/read:0Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3:0s
2PyramidFusedNet/fem_conv0/pointwise_weights/read:0=while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enter:0Rwhile/Enter:0Rwhile/Enter_1:0Rwhile/Enter_2:0bК
З
while/img_shape/cond/cond_textwhile/img_shape/cond/pred_id:0while/img_shape/cond/switch_t:0 *б
while/TensorArrayReadV3:0
#while/img_shape/cond/Shape/Switch:1
while/img_shape/cond/Shape:0
while/img_shape/cond/pred_id:0
*while/img_shape/cond/strided_slice/stack:0
,while/img_shape/cond/strided_slice/stack_1:0
,while/img_shape/cond/strided_slice/stack_2:0
$while/img_shape/cond/strided_slice:0
while/img_shape/cond/switch_t:0@
while/img_shape/cond/pred_id:0while/img_shape/cond/pred_id:0@
while/TensorArrayReadV3:0#while/img_shape/cond/Shape/Switch:1bШ
Х
 while/img_shape/cond/cond_text_1while/img_shape/cond/pred_id:0while/img_shape/cond/switch_f:0*п
while/TensorArrayReadV3:0
%while/img_shape/cond/Shape_1/Switch:0
while/img_shape/cond/Shape_1:0
while/img_shape/cond/pred_id:0
,while/img_shape/cond/strided_slice_1/stack:0
.while/img_shape/cond/strided_slice_1/stack_1:0
.while/img_shape/cond/strided_slice_1/stack_2:0
&while/img_shape/cond/strided_slice_1:0
while/img_shape/cond/switch_f:0@
while/img_shape/cond/pred_id:0while/img_shape/cond/pred_id:0B
while/TensorArrayReadV3:0%while/img_shape/cond/Shape_1/Switch:0bж
г
 while/img_shape_1/cond/cond_text while/img_shape_1/cond/pred_id:0!while/img_shape_1/cond/switch_t:0 *ч
while/TensorArrayReadV3:0
%while/img_shape_1/cond/Shape/Switch:1
while/img_shape_1/cond/Shape:0
 while/img_shape_1/cond/pred_id:0
,while/img_shape_1/cond/strided_slice/stack:0
.while/img_shape_1/cond/strided_slice/stack_1:0
.while/img_shape_1/cond/strided_slice/stack_2:0
&while/img_shape_1/cond/strided_slice:0
!while/img_shape_1/cond/switch_t:0D
 while/img_shape_1/cond/pred_id:0 while/img_shape_1/cond/pred_id:0B
while/TensorArrayReadV3:0%while/img_shape_1/cond/Shape/Switch:1bф
с
"while/img_shape_1/cond/cond_text_1 while/img_shape_1/cond/pred_id:0!while/img_shape_1/cond/switch_f:0*ѕ
while/TensorArrayReadV3:0
'while/img_shape_1/cond/Shape_1/Switch:0
 while/img_shape_1/cond/Shape_1:0
 while/img_shape_1/cond/pred_id:0
.while/img_shape_1/cond/strided_slice_1/stack:0
0while/img_shape_1/cond/strided_slice_1/stack_1:0
0while/img_shape_1/cond/strided_slice_1/stack_2:0
(while/img_shape_1/cond/strided_slice_1:0
!while/img_shape_1/cond/switch_f:0D
while/TensorArrayReadV3:0'while/img_shape_1/cond/Shape_1/Switch:0D
 while/img_shape_1/cond/pred_id:0 while/img_shape_1/cond/pred_id:0bж
г
 while/img_shape_2/cond/cond_text while/img_shape_2/cond/pred_id:0!while/img_shape_2/cond/switch_t:0 *ч
while/TensorArrayReadV3:0
%while/img_shape_2/cond/Shape/Switch:1
while/img_shape_2/cond/Shape:0
 while/img_shape_2/cond/pred_id:0
,while/img_shape_2/cond/strided_slice/stack:0
.while/img_shape_2/cond/strided_slice/stack_1:0
.while/img_shape_2/cond/strided_slice/stack_2:0
&while/img_shape_2/cond/strided_slice:0
!while/img_shape_2/cond/switch_t:0B
while/TensorArrayReadV3:0%while/img_shape_2/cond/Shape/Switch:1D
 while/img_shape_2/cond/pred_id:0 while/img_shape_2/cond/pred_id:0bф
с
"while/img_shape_2/cond/cond_text_1 while/img_shape_2/cond/pred_id:0!while/img_shape_2/cond/switch_f:0*ѕ
while/TensorArrayReadV3:0
'while/img_shape_2/cond/Shape_1/Switch:0
 while/img_shape_2/cond/Shape_1:0
 while/img_shape_2/cond/pred_id:0
.while/img_shape_2/cond/strided_slice_1/stack:0
0while/img_shape_2/cond/strided_slice_1/stack_1:0
0while/img_shape_2/cond/strided_slice_1/stack_2:0
(while/img_shape_2/cond/strided_slice_1:0
!while/img_shape_2/cond/switch_f:0D
while/TensorArrayReadV3:0'while/img_shape_2/cond/Shape_1/Switch:0D
 while/img_shape_2/cond/pred_id:0 while/img_shape_2/cond/pred_id:0bЏ
Ћ
while/cond/cond_textwhile/cond/pred_id:0while/cond/switch_t:0 *т
4PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/read:0
5PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/read:0
;PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem0_log_conv0/depthwise_weights/read:0
7PyramidFusedNet/dem0_log_conv0/pointwise_weights/read:0
4PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/read:0
5PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/read:0
;PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem0_log_conv1/depthwise_weights/read:0
7PyramidFusedNet/dem0_log_conv1/pointwise_weights/read:0
4PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/read:0
5PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/read:0
;PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem0_reg_conv0/depthwise_weights/read:0
7PyramidFusedNet/dem0_reg_conv0/pointwise_weights/read:0
4PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/read:0
5PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/read:0
;PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem0_reg_conv1/depthwise_weights/read:0
7PyramidFusedNet/dem0_reg_conv1/pointwise_weights/read:0
4PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_log_conv0/depthwise_weights/read:0
7PyramidFusedNet/dem1_log_conv0/pointwise_weights/read:0
4PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_log_conv1/depthwise_weights/read:0
7PyramidFusedNet/dem1_log_conv1/pointwise_weights/read:0
4PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_reg_conv0/depthwise_weights/read:0
7PyramidFusedNet/dem1_reg_conv0/pointwise_weights/read:0
4PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_reg_conv1/depthwise_weights/read:0
7PyramidFusedNet/dem1_reg_conv1/pointwise_weights/read:0
/PyramidFusedNet/fem_conv0/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv0/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv0/depthwise_weights/read:0
2PyramidFusedNet/fem_conv0/pointwise_weights/read:0
/PyramidFusedNet/fem_conv1/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv1/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv1/depthwise_weights/read:0
2PyramidFusedNet/fem_conv1/pointwise_weights/read:0
/PyramidFusedNet/fem_conv2/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv2/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv2/depthwise_weights/read:0
2PyramidFusedNet/fem_conv2/pointwise_weights/read:0
/PyramidFusedNet/fem_conv3/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv3/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv3/depthwise_weights/read:0
2PyramidFusedNet/fem_conv3/pointwise_weights/read:0
/PyramidFusedNet/fem_conv4/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv4/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv4/depthwise_weights/read:0
2PyramidFusedNet/fem_conv4/pointwise_weights/read:0
while/Reshape:0
while/Reshape_1:0
while/ResizeBilinear:0
while/TensorArrayReadV3:0
6while/cond/PyramidFusedNet/fem_conv0/BatchNorm/Const:0
Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch:1
Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_1:1
Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_2:1
Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_3:1
?while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm:0
?while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm:1
?while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm:2
?while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm:3
?while/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm:4
6while/cond/PyramidFusedNet/fem_conv0/LeakyRelu/alpha:0
4while/cond/PyramidFusedNet/fem_conv0/LeakyRelu/mul:0
0while/cond/PyramidFusedNet/fem_conv0/LeakyRelu:0
=while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enter:0
=while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Shape:0
>while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Switch:1
Gwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Switch:1
Jwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Switch_1:1
Awhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise:0
Ewhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/dilation_rate:0
7while/cond/PyramidFusedNet/fem_conv0/separable_conv2d:0
6while/cond/PyramidFusedNet/fem_conv1/BatchNorm/Const:0
Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch:1
Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_1:1
Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_2:1
Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_3:1
?while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm:0
?while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm:1
?while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm:2
?while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm:3
?while/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm:4
6while/cond/PyramidFusedNet/fem_conv1/LeakyRelu/alpha:0
4while/cond/PyramidFusedNet/fem_conv1/LeakyRelu/mul:0
0while/cond/PyramidFusedNet/fem_conv1/LeakyRelu:0
=while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enter:0
=while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Shape:0
>while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Switch:1
Gwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Switch:1
Awhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise:0
Ewhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/dilation_rate:0
7while/cond/PyramidFusedNet/fem_conv1/separable_conv2d:0
6while/cond/PyramidFusedNet/fem_conv2/BatchNorm/Const:0
Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch:1
Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_1:1
Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_2:1
Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_3:1
?while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm:0
?while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm:1
?while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm:2
?while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm:3
?while/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm:4
6while/cond/PyramidFusedNet/fem_conv2/LeakyRelu/alpha:0
4while/cond/PyramidFusedNet/fem_conv2/LeakyRelu/mul:0
0while/cond/PyramidFusedNet/fem_conv2/LeakyRelu:0
=while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enter:0
=while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Shape:0
>while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Switch:1
Gwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Switch:1
Awhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise:0
Ewhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/dilation_rate:0
7while/cond/PyramidFusedNet/fem_conv2/separable_conv2d:0
6while/cond/PyramidFusedNet/fem_conv3/BatchNorm/Const:0
Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch:1
Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_1:1
Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_2:1
Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_3:1
?while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm:0
?while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm:1
?while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm:2
?while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm:3
?while/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm:4
6while/cond/PyramidFusedNet/fem_conv3/LeakyRelu/alpha:0
4while/cond/PyramidFusedNet/fem_conv3/LeakyRelu/mul:0
0while/cond/PyramidFusedNet/fem_conv3/LeakyRelu:0
=while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enter:0
=while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Shape:0
>while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Switch:1
Gwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Switch:1
Awhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise:0
Ewhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/dilation_rate:0
7while/cond/PyramidFusedNet/fem_conv3/separable_conv2d:0
6while/cond/PyramidFusedNet/fem_conv4/BatchNorm/Const:0
Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3:0
Fwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch:1
Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_1:1
Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_2:1
Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_3:1
?while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm:0
?while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm:1
?while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm:2
?while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm:3
?while/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm:4
6while/cond/PyramidFusedNet/fem_conv4/LeakyRelu/alpha:0
4while/cond/PyramidFusedNet/fem_conv4/LeakyRelu/mul:0
0while/cond/PyramidFusedNet/fem_conv4/LeakyRelu:0
=while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enter:0
=while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Shape:0
>while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Switch:1
Gwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enter:0
Hwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Switch:1
Awhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise:0
Ewhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/dilation_rate:0
7while/cond/PyramidFusedNet/fem_conv4/separable_conv2d:0
8while/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_1/fem_conv0/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_1/fem_conv0/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_1/fem_conv0/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_1/fem_conv0/LeakyRelu:0
?while/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/Shape:0
Jwhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/depthwise/Switch:1
Cwhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d:0
8while/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_1/fem_conv1/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_1/fem_conv1/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_1/fem_conv1/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_1/fem_conv1/LeakyRelu:0
?while/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_1/fem_conv1/separable_conv2d:0
8while/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_1/fem_conv2/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_1/fem_conv2/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_1/fem_conv2/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_1/fem_conv2/LeakyRelu:0
?while/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_1/fem_conv2/separable_conv2d:0
8while/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_1/fem_conv3/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_1/fem_conv3/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_1/fem_conv3/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_1/fem_conv3/LeakyRelu:0
?while/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_1/fem_conv3/separable_conv2d:0
8while/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_1/fem_conv4/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_1/fem_conv4/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_1/fem_conv4/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_1/fem_conv4/LeakyRelu:0
?while/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_1/fem_conv4/separable_conv2d:0
=while/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Mwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch:1
Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_1:1
Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_2:1
Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_3:1
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_2/dem0_log_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Switch:1
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/Enter:0
Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/Switch:1
Hwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Mwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch:1
Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_1:1
Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_2:1
Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_3:1
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Switch:1
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/Enter:0
Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/Switch:1
Hwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d:0
=while/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch:1
Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1:1
Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2:1
Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3:1
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_2/dem0_reg_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Switch:1
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/Enter:0
Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/Switch:1
Hwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch:1
Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1:1
Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2:1
Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3:1
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Switch:1
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/Enter:0
Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/Switch:1
Hwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d:0
4while/cond/PyramidFusedNet_2/softmax/Reshape/shape:0
.while/cond/PyramidFusedNet_2/softmax/Reshape:0
0while/cond/PyramidFusedNet_2/softmax/Reshape_1:0
,while/cond/PyramidFusedNet_2/softmax/Shape:0
.while/cond/PyramidFusedNet_2/softmax/Softmax:0
2while/cond/PyramidFusedNet_2/strided_slice/stack:0
4while/cond/PyramidFusedNet_2/strided_slice/stack_1:0
4while/cond/PyramidFusedNet_2/strided_slice/stack_2:0
,while/cond/PyramidFusedNet_2/strided_slice:0
=while/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Mwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch:1
Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_1:1
Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_2:1
Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_3:1
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_3/dem1_log_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Switch:1
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enter:0
Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Switch:1
Hwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Mwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch:1
Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_1:1
Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_2:1
Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_3:1
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Switch:1
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enter:0
Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Switch:1
Hwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d:0
=while/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch:1
Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1:1
Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2:1
Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3:1
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_3/dem1_reg_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Switch:1
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enter:0
Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Switch:1
Hwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/Const:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch:1
Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1:1
Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2:1
Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3:1
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enter:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Switch:1
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enter:0
Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Switch:1
Hwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d:0
4while/cond/PyramidFusedNet_3/softmax/Reshape/shape:0
.while/cond/PyramidFusedNet_3/softmax/Reshape:0
0while/cond/PyramidFusedNet_3/softmax/Reshape_1:0
,while/cond/PyramidFusedNet_3/softmax/Shape:0
.while/cond/PyramidFusedNet_3/softmax/Softmax:0
2while/cond/PyramidFusedNet_3/strided_slice/stack:0
4while/cond/PyramidFusedNet_3/strided_slice/stack_1:0
4while/cond/PyramidFusedNet_3/strided_slice/stack_2:0
,while/cond/PyramidFusedNet_3/strided_slice:0
$while/cond/batch_decode/Decode/Exp:0
&while/cond/batch_decode/Decode/Exp_1:0
$while/cond/batch_decode/Decode/add:0
&while/cond/batch_decode/Decode/add_1:0
&while/cond/batch_decode/Decode/add_2:0
&while/cond/batch_decode/Decode/add_3:0
Ewhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/add:0
Gwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/add_1:0
Ewhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/sub:0
Gwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/sub_1:0
Wwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range/delta:0
Wwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range/start:0
Qwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Range:0
Wwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch:1
Pwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Rank:0
Qwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/sub/y:0
Owhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/sub:0
Qwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/sub_1:0
Kwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose:0
Kwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv/y:0
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv:0
Mwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv_1/y:0
Kwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/truediv_1:0
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:0
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:1
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:2
Iwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/unstack:3
$while/cond/batch_decode/Decode/mul:0
&while/cond/batch_decode/Decode/mul_1:0
&while/cond/batch_decode/Decode/mul_2:0
&while/cond/batch_decode/Decode/mul_3:0
&while/cond/batch_decode/Decode/stack:0
$while/cond/batch_decode/Decode/sub:0
&while/cond/batch_decode/Decode/sub_1:0
6while/cond/batch_decode/Decode/transpose/Range/delta:0
6while/cond/batch_decode/Decode/transpose/Range/start:0
0while/cond/batch_decode/Decode/transpose/Range:0
/while/cond/batch_decode/Decode/transpose/Rank:0
0while/cond/batch_decode/Decode/transpose/sub/y:0
.while/cond/batch_decode/Decode/transpose/sub:0
0while/cond/batch_decode/Decode/transpose/sub_1:0
*while/cond/batch_decode/Decode/transpose:0
8while/cond/batch_decode/Decode/transpose_1/Range/delta:0
8while/cond/batch_decode/Decode/transpose_1/Range/start:0
2while/cond/batch_decode/Decode/transpose_1/Range:0
1while/cond/batch_decode/Decode/transpose_1/Rank:0
2while/cond/batch_decode/Decode/transpose_1/sub/y:0
0while/cond/batch_decode/Decode/transpose_1/sub:0
2while/cond/batch_decode/Decode/transpose_1/sub_1:0
,while/cond/batch_decode/Decode/transpose_1:0
*while/cond/batch_decode/Decode/truediv/y:0
(while/cond/batch_decode/Decode/truediv:0
,while/cond/batch_decode/Decode/truediv_1/y:0
*while/cond/batch_decode/Decode/truediv_1:0
,while/cond/batch_decode/Decode/truediv_2/y:0
*while/cond/batch_decode/Decode/truediv_2:0
,while/cond/batch_decode/Decode/truediv_3/y:0
*while/cond/batch_decode/Decode/truediv_3:0
(while/cond/batch_decode/Decode/unstack:0
(while/cond/batch_decode/Decode/unstack:1
(while/cond/batch_decode/Decode/unstack:2
(while/cond/batch_decode/Decode/unstack:3
while/cond/batch_decode/Rank:0
 while/cond/batch_decode/Rank_1:0
'while/cond/batch_decode/Reshape/shape:0
!while/cond/batch_decode/Reshape:0
)while/cond/batch_decode/Reshape_1/shape:0
#while/cond/batch_decode/Reshape_1:0
*while/cond/batch_decode/assert_equal/All:0
;while/cond/batch_decode/assert_equal/Assert/Assert/data_0:0
;while/cond/batch_decode/assert_equal/Assert/Assert/data_1:0
;while/cond/batch_decode/assert_equal/Assert/Assert/data_2:0
;while/cond/batch_decode/assert_equal/Assert/Assert/data_4:0
3while/cond/batch_decode/assert_equal/Assert/Const:0
5while/cond/batch_decode/assert_equal/Assert/Const_1:0
5while/cond/batch_decode/assert_equal/Assert/Const_2:0
5while/cond/batch_decode/assert_equal/Assert/Const_3:0
,while/cond/batch_decode/assert_equal/Const:0
,while/cond/batch_decode/assert_equal/Equal:0
(while/cond/batch_decode/assert_equal/y:0
,while/cond/batch_decode/assert_equal_1/All:0
=while/cond/batch_decode/assert_equal_1/Assert/Assert/data_0:0
=while/cond/batch_decode/assert_equal_1/Assert/Assert/data_1:0
=while/cond/batch_decode/assert_equal_1/Assert/Assert/data_2:0
=while/cond/batch_decode/assert_equal_1/Assert/Assert/data_4:0
5while/cond/batch_decode/assert_equal_1/Assert/Const:0
7while/cond/batch_decode/assert_equal_1/Assert/Const_1:0
7while/cond/batch_decode/assert_equal_1/Assert/Const_2:0
7while/cond/batch_decode/assert_equal_1/Assert/Const_3:0
.while/cond/batch_decode/assert_equal_1/Const:0
.while/cond/batch_decode/assert_equal_1/Equal:0
*while/cond/batch_decode/assert_equal_1/y:0
1while/cond/batch_decode/nms_batch/GatherV2/axis:0
,while/cond/batch_decode/nms_batch/GatherV2:0
3while/cond/batch_decode/nms_batch/GatherV2_1/axis:0
.while/cond/batch_decode/nms_batch/GatherV2_1:0
3while/cond/batch_decode/nms_batch/batch_pad/Const:0
1while/cond/batch_decode/nms_batch/batch_pad/Max:0
3while/cond/batch_decode/nms_batch/batch_pad/Shape:0
@while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Maximum/y:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Maximum:0
:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Pad:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Reshape:0
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/Shape:0
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack:0
@while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_1/1:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_1:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_2:0
@while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_3/1:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/stack_3:0
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub/y:0
:while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub:0
<while/cond/batch_decode/nms_batch/batch_pad/pad_axis/sub_1:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/unstack:0
>while/cond/batch_decode/nms_batch/batch_pad/pad_axis/unstack:1
3while/cond/batch_decode/nms_batch/batch_pad/stack:0
5while/cond/batch_decode/nms_batch/batch_pad/stack_1:0
Awhile/cond/batch_decode/nms_batch/batch_pad/strided_slice/stack:0
Cwhile/cond/batch_decode/nms_batch/batch_pad/strided_slice/stack_1:0
Cwhile/cond/batch_decode/nms_batch/batch_pad/strided_slice/stack_2:0
;while/cond/batch_decode/nms_batch/batch_pad/strided_slice:0
5while/cond/batch_decode/nms_batch/batch_pad_1/Const:0
3while/cond/batch_decode/nms_batch/batch_pad_1/Max:0
5while/cond/batch_decode/nms_batch/batch_pad_1/Shape:0
Bwhile/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Maximum/y:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Maximum:0
<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Pad:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Reshape:0
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/Shape:0
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_1:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_2:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/stack_3:0
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub/y:0
<while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub:0
>while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/sub_1:0
@while/cond/batch_decode/nms_batch/batch_pad_1/pad_axis/unstack:0
5while/cond/batch_decode/nms_batch/batch_pad_1/stack:0
7while/cond/batch_decode/nms_batch/batch_pad_1/stack_1:0
Cwhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stack:0
Ewhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stack_1:0
Ewhile/cond/batch_decode/nms_batch/batch_pad_1/strided_slice/stack_2:0
=while/cond/batch_decode/nms_batch/batch_pad_1/strided_slice:0
5while/cond/batch_decode/nms_batch/nms/GatherV2/axis:0
0while/cond/batch_decode/nms_batch/nms/GatherV2:0
/while/cond/batch_decode/nms_batch/nms/Greater:0
Kwhile/cond/batch_decode/nms_batch/nms/NonMaxSuppressionV2/max_output_size:0
;while/cond/batch_decode/nms_batch/nms/NonMaxSuppressionV2:0
5while/cond/batch_decode/nms_batch/nms/Reshape/shape:0
/while/cond/batch_decode/nms_batch/nms/Reshape:0
-while/cond/batch_decode/nms_batch/nms/Where:0
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/GatherV2/axis:0
=while/cond/batch_decode/nms_batch/nms/boolean_mask/GatherV2:0
Kwhile/cond/batch_decode/nms_batch/nms/boolean_mask/Prod/reduction_indices:0
9while/cond/batch_decode/nms_batch/nms/boolean_mask/Prod:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape_1/shape:0
>while/cond/batch_decode/nms_batch/nms/boolean_mask/Reshape_1:0
:while/cond/batch_decode/nms_batch/nms/boolean_mask/Shape:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask/Shape_1:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask/Shape_2:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask/Squeeze:0
:while/cond/batch_decode/nms_batch/nms/boolean_mask/Where:0
@while/cond/batch_decode/nms_batch/nms/boolean_mask/concat/axis:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask/concat/values_1:0
;while/cond/batch_decode/nms_batch/nms/boolean_mask/concat:0
Hwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stack:0
Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stack_1:0
Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice/stack_2:0
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice:0
Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stack:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stack_1:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1/stack_2:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_1:0
Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stack:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stack_1:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2/stack_2:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask/strided_slice_2:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/GatherV2/axis:0
?while/cond/batch_decode/nms_batch/nms/boolean_mask_1/GatherV2:0
Mwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/Prod/reduction_indices:0
;while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Prod:0
>while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape:0
Fwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape_1/shape:0
@while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Reshape_1:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Shape:0
>while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Shape_1:0
>while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Shape_2:0
>while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Squeeze:0
<while/cond/batch_decode/nms_batch/nms/boolean_mask_1/Where:0
Bwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat/axis:0
Fwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat/values_1:0
=while/cond/batch_decode/nms_batch/nms/boolean_mask_1/concat:0
Jwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stack:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stack_1:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice/stack_2:0
Dwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stack:0
Nwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1:0
Nwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2:0
Fwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_1:0
Lwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stack:0
Nwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1:0
Nwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2:0
Fwhile/cond/batch_decode/nms_batch/nms/boolean_mask_1/strided_slice_2:0
5while/cond/batch_decode/nms_batch/nms/iou_threshold:0
7while/cond/batch_decode/nms_batch/nms/score_threshold:0
7while/cond/batch_decode/nms_batch/strided_slice/stack:0
9while/cond/batch_decode/nms_batch/strided_slice/stack_1:0
9while/cond/batch_decode/nms_batch/strided_slice/stack_2:0
1while/cond/batch_decode/nms_batch/strided_slice:0
9while/cond/batch_decode/nms_batch/strided_slice_1/stack:0
;while/cond/batch_decode/nms_batch/strided_slice_1/stack_1:0
;while/cond/batch_decode/nms_batch/strided_slice_1/stack_2:0
3while/cond/batch_decode/nms_batch/strided_slice_1:0
9while/cond/batch_decode/nms_batch/strided_slice_2/stack:0
;while/cond/batch_decode/nms_batch/strided_slice_2/stack_1:0
;while/cond/batch_decode/nms_batch/strided_slice_2/stack_2:0
3while/cond/batch_decode/nms_batch/strided_slice_2:0
9while/cond/batch_decode/nms_batch/strided_slice_3/stack:0
;while/cond/batch_decode/nms_batch/strided_slice_3/stack_1:0
;while/cond/batch_decode/nms_batch/strided_slice_3/stack_2:0
3while/cond/batch_decode/nms_batch/strided_slice_3:0
while/cond/batch_decode/stack:0
!while/cond/batch_decode/unstack:0
&while/cond/batch_decode_1/Decode/Exp:0
(while/cond/batch_decode_1/Decode/Exp_1:0
&while/cond/batch_decode_1/Decode/add:0
(while/cond/batch_decode_1/Decode/add_1:0
(while/cond/batch_decode_1/Decode/add_2:0
(while/cond/batch_decode_1/Decode/add_3:0
Gwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/add:0
Iwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/add_1:0
Gwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/sub:0
Iwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/sub_1:0
Ywhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range/delta:0
Ywhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range/start:0
Swhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Range:0
Ywhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch:1
Rwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Rank:0
Swhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/sub/y:0
Qwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/sub:0
Swhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/sub_1:0
Mwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose:0
Mwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv/y:0
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv:0
Owhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv_1/y:0
Mwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/truediv_1:0
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:0
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:1
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:2
Kwhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/unstack:3
&while/cond/batch_decode_1/Decode/mul:0
(while/cond/batch_decode_1/Decode/mul_1:0
(while/cond/batch_decode_1/Decode/mul_2:0
(while/cond/batch_decode_1/Decode/mul_3:0
(while/cond/batch_decode_1/Decode/stack:0
&while/cond/batch_decode_1/Decode/sub:0
(while/cond/batch_decode_1/Decode/sub_1:0
8while/cond/batch_decode_1/Decode/transpose/Range/delta:0
8while/cond/batch_decode_1/Decode/transpose/Range/start:0
2while/cond/batch_decode_1/Decode/transpose/Range:0
1while/cond/batch_decode_1/Decode/transpose/Rank:0
2while/cond/batch_decode_1/Decode/transpose/sub/y:0
0while/cond/batch_decode_1/Decode/transpose/sub:0
2while/cond/batch_decode_1/Decode/transpose/sub_1:0
,while/cond/batch_decode_1/Decode/transpose:0
:while/cond/batch_decode_1/Decode/transpose_1/Range/delta:0
:while/cond/batch_decode_1/Decode/transpose_1/Range/start:0
4while/cond/batch_decode_1/Decode/transpose_1/Range:0
3while/cond/batch_decode_1/Decode/transpose_1/Rank:0
4while/cond/batch_decode_1/Decode/transpose_1/sub/y:0
2while/cond/batch_decode_1/Decode/transpose_1/sub:0
4while/cond/batch_decode_1/Decode/transpose_1/sub_1:0
.while/cond/batch_decode_1/Decode/transpose_1:0
,while/cond/batch_decode_1/Decode/truediv/y:0
*while/cond/batch_decode_1/Decode/truediv:0
.while/cond/batch_decode_1/Decode/truediv_1/y:0
,while/cond/batch_decode_1/Decode/truediv_1:0
.while/cond/batch_decode_1/Decode/truediv_2/y:0
,while/cond/batch_decode_1/Decode/truediv_2:0
.while/cond/batch_decode_1/Decode/truediv_3/y:0
,while/cond/batch_decode_1/Decode/truediv_3:0
*while/cond/batch_decode_1/Decode/unstack:0
*while/cond/batch_decode_1/Decode/unstack:1
*while/cond/batch_decode_1/Decode/unstack:2
*while/cond/batch_decode_1/Decode/unstack:3
 while/cond/batch_decode_1/Rank:0
"while/cond/batch_decode_1/Rank_1:0
)while/cond/batch_decode_1/Reshape/shape:0
#while/cond/batch_decode_1/Reshape:0
+while/cond/batch_decode_1/Reshape_1/shape:0
%while/cond/batch_decode_1/Reshape_1:0
,while/cond/batch_decode_1/assert_equal/All:0
=while/cond/batch_decode_1/assert_equal/Assert/Assert/data_0:0
=while/cond/batch_decode_1/assert_equal/Assert/Assert/data_1:0
=while/cond/batch_decode_1/assert_equal/Assert/Assert/data_2:0
=while/cond/batch_decode_1/assert_equal/Assert/Assert/data_4:0
5while/cond/batch_decode_1/assert_equal/Assert/Const:0
7while/cond/batch_decode_1/assert_equal/Assert/Const_1:0
7while/cond/batch_decode_1/assert_equal/Assert/Const_2:0
7while/cond/batch_decode_1/assert_equal/Assert/Const_3:0
.while/cond/batch_decode_1/assert_equal/Const:0
.while/cond/batch_decode_1/assert_equal/Equal:0
*while/cond/batch_decode_1/assert_equal/y:0
.while/cond/batch_decode_1/assert_equal_1/All:0
?while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_0:0
?while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_1:0
?while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_2:0
?while/cond/batch_decode_1/assert_equal_1/Assert/Assert/data_4:0
7while/cond/batch_decode_1/assert_equal_1/Assert/Const:0
9while/cond/batch_decode_1/assert_equal_1/Assert/Const_1:0
9while/cond/batch_decode_1/assert_equal_1/Assert/Const_2:0
9while/cond/batch_decode_1/assert_equal_1/Assert/Const_3:0
0while/cond/batch_decode_1/assert_equal_1/Const:0
0while/cond/batch_decode_1/assert_equal_1/Equal:0
,while/cond/batch_decode_1/assert_equal_1/y:0
3while/cond/batch_decode_1/nms_batch/GatherV2/axis:0
.while/cond/batch_decode_1/nms_batch/GatherV2:0
5while/cond/batch_decode_1/nms_batch/GatherV2_1/axis:0
0while/cond/batch_decode_1/nms_batch/GatherV2_1:0
5while/cond/batch_decode_1/nms_batch/batch_pad/Const:0
3while/cond/batch_decode_1/nms_batch/batch_pad/Max:0
5while/cond/batch_decode_1/nms_batch/batch_pad/Shape:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Maximum/y:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Maximum:0
<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Pad:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Reshape:0
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/Shape:0
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_1/1:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_1:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_2:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_3/1:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/stack_3:0
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub/y:0
<while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub:0
>while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/sub_1:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/unstack:0
@while/cond/batch_decode_1/nms_batch/batch_pad/pad_axis/unstack:1
5while/cond/batch_decode_1/nms_batch/batch_pad/stack:0
7while/cond/batch_decode_1/nms_batch/batch_pad/stack_1:0
Cwhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stack:0
Ewhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stack_1:0
Ewhile/cond/batch_decode_1/nms_batch/batch_pad/strided_slice/stack_2:0
=while/cond/batch_decode_1/nms_batch/batch_pad/strided_slice:0
7while/cond/batch_decode_1/nms_batch/batch_pad_1/Const:0
5while/cond/batch_decode_1/nms_batch/batch_pad_1/Max:0
7while/cond/batch_decode_1/nms_batch/batch_pad_1/Shape:0
Dwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Maximum/y:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Maximum:0
>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Pad:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Reshape:0
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/Shape:0
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_1:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_2:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/stack_3:0
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub/y:0
>while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub:0
@while/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/sub_1:0
Bwhile/cond/batch_decode_1/nms_batch/batch_pad_1/pad_axis/unstack:0
7while/cond/batch_decode_1/nms_batch/batch_pad_1/stack:0
9while/cond/batch_decode_1/nms_batch/batch_pad_1/stack_1:0
Ewhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stack:0
Gwhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stack_1:0
Gwhile/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice/stack_2:0
?while/cond/batch_decode_1/nms_batch/batch_pad_1/strided_slice:0
7while/cond/batch_decode_1/nms_batch/nms/GatherV2/axis:0
2while/cond/batch_decode_1/nms_batch/nms/GatherV2:0
1while/cond/batch_decode_1/nms_batch/nms/Greater:0
Mwhile/cond/batch_decode_1/nms_batch/nms/NonMaxSuppressionV2/max_output_size:0
=while/cond/batch_decode_1/nms_batch/nms/NonMaxSuppressionV2:0
7while/cond/batch_decode_1/nms_batch/nms/Reshape/shape:0
1while/cond/batch_decode_1/nms_batch/nms/Reshape:0
/while/cond/batch_decode_1/nms_batch/nms/Where:0
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/GatherV2/axis:0
?while/cond/batch_decode_1/nms_batch/nms/boolean_mask/GatherV2:0
Mwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/Prod/reduction_indices:0
;while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Prod:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape_1/shape:0
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Reshape_1:0
<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Shape:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Shape_1:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Shape_2:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Squeeze:0
<while/cond/batch_decode_1/nms_batch/nms/boolean_mask/Where:0
Bwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat/axis:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat/values_1:0
=while/cond/batch_decode_1/nms_batch/nms/boolean_mask/concat:0
Jwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stack:0
Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stack_1:0
Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice/stack_2:0
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice:0
Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stack:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stack_1:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1/stack_2:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_1:0
Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stack:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stack_1:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2/stack_2:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask/strided_slice_2:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/GatherV2/axis:0
Awhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/GatherV2:0
Owhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Prod/reduction_indices:0
=while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Prod:0
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape:0
Hwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape_1/shape:0
Bwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Reshape_1:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Shape:0
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Shape_1:0
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Shape_2:0
@while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Squeeze:0
>while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/Where:0
Dwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat/axis:0
Hwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat/values_1:0
?while/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/concat:0
Lwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stack:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stack_1:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice/stack_2:0
Fwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stack:0
Pwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1:0
Pwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2:0
Hwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_1:0
Nwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stack:0
Pwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1:0
Pwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2:0
Hwhile/cond/batch_decode_1/nms_batch/nms/boolean_mask_1/strided_slice_2:0
7while/cond/batch_decode_1/nms_batch/nms/iou_threshold:0
9while/cond/batch_decode_1/nms_batch/nms/score_threshold:0
9while/cond/batch_decode_1/nms_batch/strided_slice/stack:0
;while/cond/batch_decode_1/nms_batch/strided_slice/stack_1:0
;while/cond/batch_decode_1/nms_batch/strided_slice/stack_2:0
3while/cond/batch_decode_1/nms_batch/strided_slice:0
;while/cond/batch_decode_1/nms_batch/strided_slice_1/stack:0
=while/cond/batch_decode_1/nms_batch/strided_slice_1/stack_1:0
=while/cond/batch_decode_1/nms_batch/strided_slice_1/stack_2:0
5while/cond/batch_decode_1/nms_batch/strided_slice_1:0
;while/cond/batch_decode_1/nms_batch/strided_slice_2/stack:0
=while/cond/batch_decode_1/nms_batch/strided_slice_2/stack_1:0
=while/cond/batch_decode_1/nms_batch/strided_slice_2/stack_2:0
5while/cond/batch_decode_1/nms_batch/strided_slice_2:0
;while/cond/batch_decode_1/nms_batch/strided_slice_3/stack:0
=while/cond/batch_decode_1/nms_batch/strided_slice_3/stack_1:0
=while/cond/batch_decode_1/nms_batch/strided_slice_3/stack_2:0
5while/cond/batch_decode_1/nms_batch/strided_slice_3:0
!while/cond/batch_decode_1/stack:0
#while/cond/batch_decode_1/unstack:0
while/cond/concat/axis:0
while/cond/concat:0
while/cond/concat_1/axis:0
while/cond/concat_1:0
while/cond/concat_2/axis:0
while/cond/concat_2:0
while/cond/pred_id:0
while/cond/switch_t:0
:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/read:0Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_3:1
Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter:0Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter:0
Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0
5PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch:1
:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/read:0Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_3:1
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2:0Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2:0Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2:0
Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter:0Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter:0
;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_2:1
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2:0Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2:0Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2:0{
/PyramidFusedNet/fem_conv2/BatchNorm/beta/read:0Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_1:1
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2:0Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enter:0
?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3:1t
2PyramidFusedNet/fem_conv4/pointwise_weights/read:0>while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Switch:1
4PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_1:1 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enter:0Gwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enter:0~
2PyramidFusedNet/fem_conv4/depthwise_weights/read:0Hwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Switch:1
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3:0Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enter:0Gwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3:0Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3:0 
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
?PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_3:1
Gwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enter:0Gwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enter:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0 
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3:0Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3:0
Gwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enter:0Gwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enter:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3:0Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Gwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enter:0Gwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3:0Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
?PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3:1
;PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2:1
;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2:1 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0d
while/ResizeBilinear:0Jwhile/cond/PyramidFusedNet_1/fem_conv0/separable_conv2d/depthwise/Switch:1 
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0~
=while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enter:0=while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enter:0 
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
4PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1:1
5PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch:1
6PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/read:0Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_2:1
6PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/read:0Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_2:1{
/PyramidFusedNet/fem_conv1/BatchNorm/beta/read:0Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_1:1
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enter:0n
while/Reshape_1:0Ywhile/cond/batch_decode_1/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch:1 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0 
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/Enter:0
4PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_1:1 
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0 
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/Enter:0
5PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch:1 
Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
;PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_2:1
;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_2:1~
=while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enter:0=while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enter:0z
0PyramidFusedNet/fem_conv4/BatchNorm/gamma/read:0Fwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch:1,
while/cond/pred_id:0while/cond/pred_id:0
Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter:0
5PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch:1
4PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1:1
:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/read:0Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_3:1z
0PyramidFusedNet/fem_conv3/BatchNorm/gamma/read:0Fwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch:1
Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter:0Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter:0{
/PyramidFusedNet/fem_conv0/BatchNorm/beta/read:0Hwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch_1:1
?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_3:1
:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/read:0Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_3:1t
2PyramidFusedNet/fem_conv0/pointwise_weights/read:0>while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Switch:1
Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter:0Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter:0
Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0
7PyramidFusedNet/dem1_log_conv0/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Switch:1~
=while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enter:0=while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enter:0
;PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2:1
7PyramidFusedNet/dem0_log_conv0/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Switch:1
2PyramidFusedNet/fem_conv0/depthwise_weights/read:0Jwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Switch_1:1
5PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch:1
:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/read:0Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_3:1z
0PyramidFusedNet/fem_conv2/BatchNorm/gamma/read:0Fwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch:1
7PyramidFusedNet/dem1_log_conv0/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Switch:1
?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3:1
7PyramidFusedNet/dem0_log_conv0/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/Switch:1
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
7PyramidFusedNet/dem1_reg_conv0/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Switch:1
6PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/read:0Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_2:1
6PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/read:0Hwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch_2:1
4PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_1:1
7PyramidFusedNet/dem0_reg_conv0/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Switch:1
?PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch_3:1z
0PyramidFusedNet/fem_conv1/BatchNorm/gamma/read:0Fwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Switch:1
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0
7PyramidFusedNet/dem1_reg_conv0/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Switch:1{
/PyramidFusedNet/fem_conv4/BatchNorm/beta/read:0Hwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Switch_1:1
7PyramidFusedNet/dem0_reg_conv0/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/depthwise/Switch:1t
2PyramidFusedNet/fem_conv1/pointwise_weights/read:0>while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Switch:1 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
7PyramidFusedNet/dem1_log_conv1/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Switch:1~
=while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enter:0=while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0 
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
7PyramidFusedNet/dem0_log_conv1/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Switch:1~
2PyramidFusedNet/fem_conv1/depthwise_weights/read:0Hwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Switch:1 
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0z
0PyramidFusedNet/fem_conv0/BatchNorm/gamma/read:0Fwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Switch:1
7PyramidFusedNet/dem1_log_conv1/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Switch:1
;PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch_2:1
7PyramidFusedNet/dem0_log_conv1/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/Switch:1
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/Enter:0e
while/TensorArrayReadV3:0Hwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Switch:1
?PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3:1
5PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Switch:1
4PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1:1
7PyramidFusedNet/dem1_reg_conv1/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Switch:1 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enter:0
7PyramidFusedNet/dem0_reg_conv1/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Switch:1 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
7PyramidFusedNet/dem1_reg_conv1/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Switch:1 
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv0/separable_conv2d/Enter:0
7PyramidFusedNet/dem0_reg_conv1/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/depthwise/Switch:1 
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0t
2PyramidFusedNet/fem_conv2/pointwise_weights/read:0>while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Switch:1
5PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Switch:1~
=while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enter:0=while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enter:0~
2PyramidFusedNet/fem_conv2/depthwise_weights/read:0Hwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Switch:1
4PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_1:1
Dwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/Enter:0{
/PyramidFusedNet/fem_conv3/BatchNorm/beta/read:0Hwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Switch_1:1 
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/separable_conv2d/depthwise/Enter:0
;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2:1 
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0 
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/separable_conv2d/depthwise/Enter:0 
Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_2/dem0_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0j
while/Reshape:0Wwhile/cond/batch_decode/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch:1
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1:0Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1:0
Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/separable_conv2d/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1:0Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1:0
6PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/read:0Hwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Switch_2:1t
2PyramidFusedNet/fem_conv3/pointwise_weights/read:0>while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Switch:1
5PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_2/dem0_reg_conv1/BatchNorm/FusedBatchNorm/Switch:1
Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_2/dem0_log_conv1/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1:0Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1:0
Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter:0Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1:0Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
4PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_2/dem0_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1:1~
2PyramidFusedNet/fem_conv3/depthwise_weights/read:0Hwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Switch:1
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1:0Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_3:1b

while/cond/cond_text_1while/cond/pred_id:0while/cond/switch_f:0*Ъ
4PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_log_conv0/depthwise_weights/read:0
7PyramidFusedNet/dem1_log_conv0/pointwise_weights/read:0
4PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_log_conv1/depthwise_weights/read:0
7PyramidFusedNet/dem1_log_conv1/pointwise_weights/read:0
4PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_reg_conv0/depthwise_weights/read:0
7PyramidFusedNet/dem1_reg_conv0/pointwise_weights/read:0
4PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/read:0
5PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/read:0
;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/read:0
?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/read:0
7PyramidFusedNet/dem1_reg_conv1/depthwise_weights/read:0
7PyramidFusedNet/dem1_reg_conv1/pointwise_weights/read:0
/PyramidFusedNet/fem_conv0/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv0/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv0/depthwise_weights/read:0
2PyramidFusedNet/fem_conv0/pointwise_weights/read:0
/PyramidFusedNet/fem_conv1/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv1/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv1/depthwise_weights/read:0
2PyramidFusedNet/fem_conv1/pointwise_weights/read:0
/PyramidFusedNet/fem_conv2/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv2/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv2/depthwise_weights/read:0
2PyramidFusedNet/fem_conv2/pointwise_weights/read:0
/PyramidFusedNet/fem_conv3/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv3/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv3/depthwise_weights/read:0
2PyramidFusedNet/fem_conv3/pointwise_weights/read:0
/PyramidFusedNet/fem_conv4/BatchNorm/beta/read:0
0PyramidFusedNet/fem_conv4/BatchNorm/gamma/read:0
6PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/read:0
:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/read:0
2PyramidFusedNet/fem_conv4/depthwise_weights/read:0
2PyramidFusedNet/fem_conv4/pointwise_weights/read:0
while/Reshape_1:0
while/ResizeBilinear:0
while/TensorArrayReadV3:0
Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
=while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enter:0
Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
=while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enter:0
Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3:0
=while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enter:0
Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3:0
=while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enter:0
Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3:0
=while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enter:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enter:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enter:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enter:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enter:0
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enter:0
8while/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/Const:0
Hwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch:0
Jwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_1:0
Jwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_2:0
Jwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_3:0
Awhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_4/fem_conv0/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_4/fem_conv0/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_4/fem_conv0/LeakyRelu:0
?while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/Shape:0
@while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/Switch:0
Jwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise/Switch:0
Lwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise/Switch_1:0
Cwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d:0
8while/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/Const:0
Hwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch:0
Jwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_1:0
Jwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_2:0
Jwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_3:0
Awhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_4/fem_conv1/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_4/fem_conv1/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_4/fem_conv1/LeakyRelu:0
?while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/Shape:0
@while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/Switch:0
Jwhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/depthwise/Switch:0
Cwhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d:0
8while/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/Const:0
Hwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch:0
Jwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_1:0
Jwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_2:0
Jwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_3:0
Awhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_4/fem_conv2/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_4/fem_conv2/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_4/fem_conv2/LeakyRelu:0
?while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/Shape:0
@while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/Switch:0
Jwhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/depthwise/Switch:0
Cwhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d:0
8while/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/Const:0
Hwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch:0
Jwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_1:0
Jwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_2:0
Jwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_3:0
Awhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_4/fem_conv3/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_4/fem_conv3/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_4/fem_conv3/LeakyRelu:0
?while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/Shape:0
@while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/Switch:0
Jwhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/depthwise/Switch:0
Cwhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d:0
8while/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/Const:0
Hwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch:0
Jwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_1:0
Jwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_2:0
Jwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_3:0
Awhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_4/fem_conv4/LeakyRelu:0
?while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/Shape:0
@while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/Switch:0
Jwhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/depthwise/Switch:0
Cwhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d:0
8while/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_5/fem_conv0/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_5/fem_conv0/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_5/fem_conv0/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_5/fem_conv0/LeakyRelu:0
?while/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/Shape:0
Jwhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/depthwise/Switch:0
Cwhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d:0
8while/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_5/fem_conv1/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_5/fem_conv1/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_5/fem_conv1/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_5/fem_conv1/LeakyRelu:0
?while/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_5/fem_conv1/separable_conv2d:0
8while/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_5/fem_conv2/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_5/fem_conv2/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_5/fem_conv2/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_5/fem_conv2/LeakyRelu:0
?while/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_5/fem_conv2/separable_conv2d:0
8while/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_5/fem_conv3/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_5/fem_conv3/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_5/fem_conv3/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_5/fem_conv3/LeakyRelu:0
?while/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_5/fem_conv3/separable_conv2d:0
8while/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/Const:0
Awhile/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm:0
Awhile/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm:1
Awhile/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm:2
Awhile/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm:3
Awhile/cond/PyramidFusedNet_5/fem_conv4/BatchNorm/FusedBatchNorm:4
8while/cond/PyramidFusedNet_5/fem_conv4/LeakyRelu/alpha:0
6while/cond/PyramidFusedNet_5/fem_conv4/LeakyRelu/mul:0
2while/cond/PyramidFusedNet_5/fem_conv4/LeakyRelu:0
?while/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d/Shape:0
Cwhile/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d/depthwise:0
Gwhile/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d/dilation_rate:0
9while/cond/PyramidFusedNet_5/fem_conv4/separable_conv2d:0
=while/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/Const:0
Mwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch:0
Owhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_1:0
Owhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_2:0
Owhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_3:0
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_6/dem1_log_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/Switch:0
Owhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/depthwise/Switch:0
Hwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/Const:0
Mwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch:0
Owhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_1:0
Owhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_2:0
Owhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_3:0
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/Switch:0
Owhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/depthwise/Switch:0
Hwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d:0
=while/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/Const:0
Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch:0
Owhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1:0
Owhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2:0
Owhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3:0
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm:4
=while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyRelu/alpha:0
;while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyRelu/mul:0
7while/cond/PyramidFusedNet_6/dem1_reg_conv0/LeakyRelu:0
Dwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/Switch:0
Owhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/depthwise/Switch:0
Hwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d:0
=while/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/Const:0
Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch:0
Owhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1:0
Owhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2:0
Owhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3:0
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm:0
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm:1
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm:2
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm:3
Fwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm:4
Dwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/Shape:0
Ewhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/Switch:0
Owhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/depthwise/Switch:0
Hwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/depthwise:0
Lwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/dilation_rate:0
>while/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d:0
4while/cond/PyramidFusedNet_6/softmax/Reshape/shape:0
.while/cond/PyramidFusedNet_6/softmax/Reshape:0
0while/cond/PyramidFusedNet_6/softmax/Reshape_1:0
,while/cond/PyramidFusedNet_6/softmax/Shape:0
.while/cond/PyramidFusedNet_6/softmax/Softmax:0
2while/cond/PyramidFusedNet_6/strided_slice/stack:0
4while/cond/PyramidFusedNet_6/strided_slice/stack_1:0
4while/cond/PyramidFusedNet_6/strided_slice/stack_2:0
,while/cond/PyramidFusedNet_6/strided_slice:0
&while/cond/batch_decode_2/Decode/Exp:0
(while/cond/batch_decode_2/Decode/Exp_1:0
&while/cond/batch_decode_2/Decode/add:0
(while/cond/batch_decode_2/Decode/add_1:0
(while/cond/batch_decode_2/Decode/add_2:0
(while/cond/batch_decode_2/Decode/add_3:0
Gwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/add:0
Iwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/add_1:0
Gwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/sub:0
Iwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/sub_1:0
Ywhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range/delta:0
Ywhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range/start:0
Swhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Range:0
Ywhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch:0
Rwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Rank:0
Swhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/sub/y:0
Qwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/sub:0
Swhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/sub_1:0
Mwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose:0
Mwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv/y:0
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv:0
Owhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv_1/y:0
Mwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/truediv_1:0
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:0
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:1
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:2
Kwhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/unstack:3
&while/cond/batch_decode_2/Decode/mul:0
(while/cond/batch_decode_2/Decode/mul_1:0
(while/cond/batch_decode_2/Decode/mul_2:0
(while/cond/batch_decode_2/Decode/mul_3:0
(while/cond/batch_decode_2/Decode/stack:0
&while/cond/batch_decode_2/Decode/sub:0
(while/cond/batch_decode_2/Decode/sub_1:0
8while/cond/batch_decode_2/Decode/transpose/Range/delta:0
8while/cond/batch_decode_2/Decode/transpose/Range/start:0
2while/cond/batch_decode_2/Decode/transpose/Range:0
1while/cond/batch_decode_2/Decode/transpose/Rank:0
2while/cond/batch_decode_2/Decode/transpose/sub/y:0
0while/cond/batch_decode_2/Decode/transpose/sub:0
2while/cond/batch_decode_2/Decode/transpose/sub_1:0
,while/cond/batch_decode_2/Decode/transpose:0
:while/cond/batch_decode_2/Decode/transpose_1/Range/delta:0
:while/cond/batch_decode_2/Decode/transpose_1/Range/start:0
4while/cond/batch_decode_2/Decode/transpose_1/Range:0
3while/cond/batch_decode_2/Decode/transpose_1/Rank:0
4while/cond/batch_decode_2/Decode/transpose_1/sub/y:0
2while/cond/batch_decode_2/Decode/transpose_1/sub:0
4while/cond/batch_decode_2/Decode/transpose_1/sub_1:0
.while/cond/batch_decode_2/Decode/transpose_1:0
,while/cond/batch_decode_2/Decode/truediv/y:0
*while/cond/batch_decode_2/Decode/truediv:0
.while/cond/batch_decode_2/Decode/truediv_1/y:0
,while/cond/batch_decode_2/Decode/truediv_1:0
.while/cond/batch_decode_2/Decode/truediv_2/y:0
,while/cond/batch_decode_2/Decode/truediv_2:0
.while/cond/batch_decode_2/Decode/truediv_3/y:0
,while/cond/batch_decode_2/Decode/truediv_3:0
*while/cond/batch_decode_2/Decode/unstack:0
*while/cond/batch_decode_2/Decode/unstack:1
*while/cond/batch_decode_2/Decode/unstack:2
*while/cond/batch_decode_2/Decode/unstack:3
 while/cond/batch_decode_2/Rank:0
"while/cond/batch_decode_2/Rank_1:0
)while/cond/batch_decode_2/Reshape/shape:0
#while/cond/batch_decode_2/Reshape:0
+while/cond/batch_decode_2/Reshape_1/shape:0
%while/cond/batch_decode_2/Reshape_1:0
,while/cond/batch_decode_2/assert_equal/All:0
=while/cond/batch_decode_2/assert_equal/Assert/Assert/data_0:0
=while/cond/batch_decode_2/assert_equal/Assert/Assert/data_1:0
=while/cond/batch_decode_2/assert_equal/Assert/Assert/data_2:0
=while/cond/batch_decode_2/assert_equal/Assert/Assert/data_4:0
5while/cond/batch_decode_2/assert_equal/Assert/Const:0
7while/cond/batch_decode_2/assert_equal/Assert/Const_1:0
7while/cond/batch_decode_2/assert_equal/Assert/Const_2:0
7while/cond/batch_decode_2/assert_equal/Assert/Const_3:0
.while/cond/batch_decode_2/assert_equal/Const:0
.while/cond/batch_decode_2/assert_equal/Equal:0
*while/cond/batch_decode_2/assert_equal/y:0
.while/cond/batch_decode_2/assert_equal_1/All:0
?while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_0:0
?while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_1:0
?while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_2:0
?while/cond/batch_decode_2/assert_equal_1/Assert/Assert/data_4:0
7while/cond/batch_decode_2/assert_equal_1/Assert/Const:0
9while/cond/batch_decode_2/assert_equal_1/Assert/Const_1:0
9while/cond/batch_decode_2/assert_equal_1/Assert/Const_2:0
9while/cond/batch_decode_2/assert_equal_1/Assert/Const_3:0
0while/cond/batch_decode_2/assert_equal_1/Const:0
0while/cond/batch_decode_2/assert_equal_1/Equal:0
,while/cond/batch_decode_2/assert_equal_1/y:0
3while/cond/batch_decode_2/nms_batch/GatherV2/axis:0
.while/cond/batch_decode_2/nms_batch/GatherV2:0
5while/cond/batch_decode_2/nms_batch/GatherV2_1/axis:0
0while/cond/batch_decode_2/nms_batch/GatherV2_1:0
5while/cond/batch_decode_2/nms_batch/batch_pad/Const:0
3while/cond/batch_decode_2/nms_batch/batch_pad/Max:0
5while/cond/batch_decode_2/nms_batch/batch_pad/Shape:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Maximum/y:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Maximum:0
<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Pad:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Reshape:0
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/Shape:0
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_1/1:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_1:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_2:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_3/1:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/stack_3:0
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub/y:0
<while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub:0
>while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/sub_1:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/unstack:0
@while/cond/batch_decode_2/nms_batch/batch_pad/pad_axis/unstack:1
5while/cond/batch_decode_2/nms_batch/batch_pad/stack:0
7while/cond/batch_decode_2/nms_batch/batch_pad/stack_1:0
Cwhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stack:0
Ewhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stack_1:0
Ewhile/cond/batch_decode_2/nms_batch/batch_pad/strided_slice/stack_2:0
=while/cond/batch_decode_2/nms_batch/batch_pad/strided_slice:0
7while/cond/batch_decode_2/nms_batch/batch_pad_1/Const:0
5while/cond/batch_decode_2/nms_batch/batch_pad_1/Max:0
7while/cond/batch_decode_2/nms_batch/batch_pad_1/Shape:0
Dwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Maximum/y:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Maximum:0
>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Pad:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Reshape:0
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/Shape:0
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_1:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_2:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/stack_3:0
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub/y:0
>while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub:0
@while/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/sub_1:0
Bwhile/cond/batch_decode_2/nms_batch/batch_pad_1/pad_axis/unstack:0
7while/cond/batch_decode_2/nms_batch/batch_pad_1/stack:0
9while/cond/batch_decode_2/nms_batch/batch_pad_1/stack_1:0
Ewhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stack:0
Gwhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stack_1:0
Gwhile/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice/stack_2:0
?while/cond/batch_decode_2/nms_batch/batch_pad_1/strided_slice:0
7while/cond/batch_decode_2/nms_batch/nms/GatherV2/axis:0
2while/cond/batch_decode_2/nms_batch/nms/GatherV2:0
1while/cond/batch_decode_2/nms_batch/nms/Greater:0
Mwhile/cond/batch_decode_2/nms_batch/nms/NonMaxSuppressionV2/max_output_size:0
=while/cond/batch_decode_2/nms_batch/nms/NonMaxSuppressionV2:0
7while/cond/batch_decode_2/nms_batch/nms/Reshape/shape:0
1while/cond/batch_decode_2/nms_batch/nms/Reshape:0
/while/cond/batch_decode_2/nms_batch/nms/Where:0
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/GatherV2/axis:0
?while/cond/batch_decode_2/nms_batch/nms/boolean_mask/GatherV2:0
Mwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/Prod/reduction_indices:0
;while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Prod:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape_1/shape:0
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Reshape_1:0
<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Shape:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Shape_1:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Shape_2:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Squeeze:0
<while/cond/batch_decode_2/nms_batch/nms/boolean_mask/Where:0
Bwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat/axis:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat/values_1:0
=while/cond/batch_decode_2/nms_batch/nms/boolean_mask/concat:0
Jwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stack:0
Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stack_1:0
Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice/stack_2:0
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice:0
Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stack:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stack_1:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1/stack_2:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_1:0
Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stack:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stack_1:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2/stack_2:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask/strided_slice_2:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/GatherV2/axis:0
Awhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/GatherV2:0
Owhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Prod/reduction_indices:0
=while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Prod:0
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape:0
Hwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape_1/shape:0
Bwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Reshape_1:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Shape:0
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Shape_1:0
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Shape_2:0
@while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Squeeze:0
>while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/Where:0
Dwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat/axis:0
Hwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat/values_1:0
?while/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/concat:0
Lwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stack:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stack_1:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice/stack_2:0
Fwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stack:0
Pwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_1:0
Pwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1/stack_2:0
Hwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_1:0
Nwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stack:0
Pwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_1:0
Pwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2/stack_2:0
Hwhile/cond/batch_decode_2/nms_batch/nms/boolean_mask_1/strided_slice_2:0
7while/cond/batch_decode_2/nms_batch/nms/iou_threshold:0
9while/cond/batch_decode_2/nms_batch/nms/score_threshold:0
9while/cond/batch_decode_2/nms_batch/strided_slice/stack:0
;while/cond/batch_decode_2/nms_batch/strided_slice/stack_1:0
;while/cond/batch_decode_2/nms_batch/strided_slice/stack_2:0
3while/cond/batch_decode_2/nms_batch/strided_slice:0
;while/cond/batch_decode_2/nms_batch/strided_slice_1/stack:0
=while/cond/batch_decode_2/nms_batch/strided_slice_1/stack_1:0
=while/cond/batch_decode_2/nms_batch/strided_slice_1/stack_2:0
5while/cond/batch_decode_2/nms_batch/strided_slice_1:0
;while/cond/batch_decode_2/nms_batch/strided_slice_2/stack:0
=while/cond/batch_decode_2/nms_batch/strided_slice_2/stack_1:0
=while/cond/batch_decode_2/nms_batch/strided_slice_2/stack_2:0
5while/cond/batch_decode_2/nms_batch/strided_slice_2:0
;while/cond/batch_decode_2/nms_batch/strided_slice_3/stack:0
=while/cond/batch_decode_2/nms_batch/strided_slice_3/stack_1:0
=while/cond/batch_decode_2/nms_batch/strided_slice_3/stack_2:0
5while/cond/batch_decode_2/nms_batch/strided_slice_3:0
!while/cond/batch_decode_2/stack:0
#while/cond/batch_decode_2/unstack:0
while/cond/concat_3/axis:0
while/cond/concat_3:0
while/cond/pred_id:0
while/cond/switch_f:0
;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_2:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_2:0d
while/ResizeBilinear:0Jwhile/cond/PyramidFusedNet_5/fem_conv0/separable_conv2d/depthwise/Switch:0~
=while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enter:0=while/cond/PyramidFusedNet/fem_conv0/separable_conv2d/Enter:0
4PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_1:0
5PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch:0
6PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_2:0
6PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_2:0}
/PyramidFusedNet/fem_conv1/BatchNorm/beta/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_1:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/separable_conv2d/depthwise/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_3:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/depthwise/Enter:0n
while/Reshape_1:0Ywhile/cond/batch_decode_2/Decode/get_center_coordinates_and_sizes/transpose/Rank/Switch:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
5PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch:0
;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_2:0~
=while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enter:0=while/cond/PyramidFusedNet/fem_conv1/separable_conv2d/Enter:0|
0PyramidFusedNet/fem_conv4/BatchNorm/gamma/read:0Hwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch:0,
while/cond/pred_id:0while/cond/pred_id:0
4PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_1:0
:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_3:0
5PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch:0|
0PyramidFusedNet/fem_conv3/BatchNorm/gamma/read:0Hwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch:0
Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter:0Ewhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter:0}
/PyramidFusedNet/fem_conv0/BatchNorm/beta/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch_1:0
?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_3:0
:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_3:0v
2PyramidFusedNet/fem_conv0/pointwise_weights/read:0@while/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/Switch:0
Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter:0Ewhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter:0
7PyramidFusedNet/dem1_log_conv0/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/Switch:0~
=while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enter:0=while/cond/PyramidFusedNet/fem_conv2/separable_conv2d/Enter:0
2PyramidFusedNet/fem_conv0/depthwise_weights/read:0Lwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise/Switch_1:0
5PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/read:0Mwhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch:0
:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_3:0|
0PyramidFusedNet/fem_conv2/BatchNorm/gamma/read:0Hwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch:0
7PyramidFusedNet/dem1_log_conv0/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_6/dem1_log_conv0/separable_conv2d/depthwise/Switch:0
?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_3:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_1:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
7PyramidFusedNet/dem1_reg_conv0/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/Switch:0
6PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_2:0
6PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_2:0
4PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_6/dem1_log_conv1/BatchNorm/FusedBatchNorm/Switch_1:0|
0PyramidFusedNet/fem_conv1/BatchNorm/gamma/read:0Hwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter:0
7PyramidFusedNet/dem1_reg_conv0/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_6/dem1_reg_conv0/separable_conv2d/depthwise/Switch:0}
/PyramidFusedNet/fem_conv4/BatchNorm/beta/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv4/BatchNorm/FusedBatchNorm/Switch_1:0v
2PyramidFusedNet/fem_conv1/pointwise_weights/read:0@while/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/Switch:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_2:0
7PyramidFusedNet/dem1_log_conv1/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/Switch:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_2:0~
=while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enter:0=while/cond/PyramidFusedNet/fem_conv3/separable_conv2d/Enter:0
2PyramidFusedNet/fem_conv1/depthwise_weights/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv1/separable_conv2d/depthwise/Switch:0|
0PyramidFusedNet/fem_conv0/BatchNorm/gamma/read:0Hwhile/cond/PyramidFusedNet_4/fem_conv0/BatchNorm/FusedBatchNorm/Switch:0
7PyramidFusedNet/dem1_log_conv1/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_6/dem1_log_conv1/separable_conv2d/depthwise/Switch:0g
while/TensorArrayReadV3:0Jwhile/cond/PyramidFusedNet_4/fem_conv0/separable_conv2d/depthwise/Switch:0
7PyramidFusedNet/dem1_reg_conv1/pointwise_weights/read:0Ewhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/Switch:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/separable_conv2d/depthwise/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter_3:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enter:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/depthwise/Enter:0 
Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0Nwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Enter_3:0
7PyramidFusedNet/dem1_reg_conv1/depthwise_weights/read:0Owhile/cond/PyramidFusedNet_6/dem1_reg_conv1/separable_conv2d/depthwise/Switch:0v
2PyramidFusedNet/fem_conv2/pointwise_weights/read:0@while/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/Switch:0~
=while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enter:0=while/cond/PyramidFusedNet/fem_conv4/separable_conv2d/Enter:0
2PyramidFusedNet/fem_conv2/depthwise_weights/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv2/separable_conv2d/depthwise/Switch:0
4PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/read:0Owhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_1:0}
/PyramidFusedNet/fem_conv3/BatchNorm/beta/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_1:0
;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_6/dem1_reg_conv0/BatchNorm/FusedBatchNorm/Switch_2:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1:0Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1:0Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_1:0v
2PyramidFusedNet/fem_conv3/pointwise_weights/read:0@while/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/Switch:0
6PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_2:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1:0Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_1:0
Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter:0Ewhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1:0Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
2PyramidFusedNet/fem_conv3/depthwise_weights/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv3/separable_conv2d/depthwise/Switch:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1:0Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_3:0
:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv1/BatchNorm/FusedBatchNorm/Switch_3:0
Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter:0Ewhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter:0
:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv3/BatchNorm/FusedBatchNorm/Switch_3:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2:0Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2:0Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_2:0
Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter:0Ewhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter:0
Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter:0
;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/read:0Owhile/cond/PyramidFusedNet_6/dem1_log_conv0/BatchNorm/FusedBatchNorm/Switch_2:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2:0Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_2:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2:0Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_2:0}
/PyramidFusedNet/fem_conv2/BatchNorm/beta/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv2/BatchNorm/FusedBatchNorm/Switch_1:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2:0Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_2:0
Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_3/dem1_log_conv0/separable_conv2d/Enter:0
?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/read:0Owhile/cond/PyramidFusedNet_6/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Switch_3:0v
2PyramidFusedNet/fem_conv4/pointwise_weights/read:0@while/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/Switch:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv1/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enter:0Gwhile/cond/PyramidFusedNet/fem_conv4/separable_conv2d/depthwise/Enter:0
2PyramidFusedNet/fem_conv4/depthwise_weights/read:0Jwhile/cond/PyramidFusedNet_4/fem_conv4/separable_conv2d/depthwise/Switch:0
Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3:0Gwhile/cond/PyramidFusedNet/fem_conv4/BatchNorm/FusedBatchNorm/Enter_3:0 
Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0Nwhile/cond/PyramidFusedNet_3/dem1_log_conv0/BatchNorm/FusedBatchNorm/Enter_1:0
Gwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enter:0Gwhile/cond/PyramidFusedNet/fem_conv3/separable_conv2d/depthwise/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3:0Gwhile/cond/PyramidFusedNet/fem_conv3/BatchNorm/FusedBatchNorm/Enter_3:0
Gwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enter:0Gwhile/cond/PyramidFusedNet/fem_conv2/separable_conv2d/depthwise/Enter:0
Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0Lwhile/cond/PyramidFusedNet_3/dem1_reg_conv1/BatchNorm/FusedBatchNorm/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3:0Gwhile/cond/PyramidFusedNet/fem_conv2/BatchNorm/FusedBatchNorm/Enter_3:0
Gwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enter:0Gwhile/cond/PyramidFusedNet/fem_conv1/separable_conv2d/depthwise/Enter:0
Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enter:0Dwhile/cond/PyramidFusedNet_3/dem1_reg_conv0/separable_conv2d/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3:0Gwhile/cond/PyramidFusedNet/fem_conv1/BatchNorm/FusedBatchNorm/Enter_3:0
Gwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enter:0Gwhile/cond/PyramidFusedNet/fem_conv0/separable_conv2d/depthwise/Enter:0
Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3:0Gwhile/cond/PyramidFusedNet/fem_conv0/BatchNorm/FusedBatchNorm/Enter_3:0bж
г
 while/img_shape_3/cond/cond_text while/img_shape_3/cond/pred_id:0!while/img_shape_3/cond/switch_t:0 *ч
while/TensorArrayReadV3:0
%while/img_shape_3/cond/Shape/Switch:1
while/img_shape_3/cond/Shape:0
 while/img_shape_3/cond/pred_id:0
,while/img_shape_3/cond/strided_slice/stack:0
.while/img_shape_3/cond/strided_slice/stack_1:0
.while/img_shape_3/cond/strided_slice/stack_2:0
&while/img_shape_3/cond/strided_slice:0
!while/img_shape_3/cond/switch_t:0D
 while/img_shape_3/cond/pred_id:0 while/img_shape_3/cond/pred_id:0B
while/TensorArrayReadV3:0%while/img_shape_3/cond/Shape/Switch:1bф
с
"while/img_shape_3/cond/cond_text_1 while/img_shape_3/cond/pred_id:0!while/img_shape_3/cond/switch_f:0*ѕ
while/TensorArrayReadV3:0
'while/img_shape_3/cond/Shape_1/Switch:0
 while/img_shape_3/cond/Shape_1:0
 while/img_shape_3/cond/pred_id:0
.while/img_shape_3/cond/strided_slice_1/stack:0
0while/img_shape_3/cond/strided_slice_1/stack_1:0
0while/img_shape_3/cond/strided_slice_1/stack_2:0
(while/img_shape_3/cond/strided_slice_1:0
!while/img_shape_3/cond/switch_f:0D
 while/img_shape_3/cond/pred_id:0 while/img_shape_3/cond/pred_id:0D
while/TensorArrayReadV3:0'while/img_shape_3/cond/Shape_1/Switch:0"
	variablesќј
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
у
-PyramidFusedNet/fem_conv0/depthwise_weights:02PyramidFusedNet/fem_conv0/depthwise_weights/Assign2PyramidFusedNet/fem_conv0/depthwise_weights/read:02HPyramidFusedNet/fem_conv0/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv0/pointwise_weights:02PyramidFusedNet/fem_conv0/pointwise_weights/Assign2PyramidFusedNet/fem_conv0/pointwise_weights/read:02HPyramidFusedNet/fem_conv0/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv0/BatchNorm/gamma:00PyramidFusedNet/fem_conv0/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv0/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv0/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv0/BatchNorm/beta:0/PyramidFusedNet/fem_conv0/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv0/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv0/BatchNorm/beta/Initializer/zeros:08
ш
1PyramidFusedNet/fem_conv0/BatchNorm/moving_mean:06PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/Assign6PyramidFusedNet/fem_conv0/BatchNorm/moving_mean/read:02CPyramidFusedNet/fem_conv0/BatchNorm/moving_mean/Initializer/zeros:0
ї
5PyramidFusedNet/fem_conv0/BatchNorm/moving_variance:0:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/Assign:PyramidFusedNet/fem_conv0/BatchNorm/moving_variance/read:02FPyramidFusedNet/fem_conv0/BatchNorm/moving_variance/Initializer/ones:0
у
-PyramidFusedNet/fem_conv1/depthwise_weights:02PyramidFusedNet/fem_conv1/depthwise_weights/Assign2PyramidFusedNet/fem_conv1/depthwise_weights/read:02HPyramidFusedNet/fem_conv1/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv1/pointwise_weights:02PyramidFusedNet/fem_conv1/pointwise_weights/Assign2PyramidFusedNet/fem_conv1/pointwise_weights/read:02HPyramidFusedNet/fem_conv1/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv1/BatchNorm/gamma:00PyramidFusedNet/fem_conv1/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv1/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv1/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv1/BatchNorm/beta:0/PyramidFusedNet/fem_conv1/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv1/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv1/BatchNorm/beta/Initializer/zeros:08
ш
1PyramidFusedNet/fem_conv1/BatchNorm/moving_mean:06PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/Assign6PyramidFusedNet/fem_conv1/BatchNorm/moving_mean/read:02CPyramidFusedNet/fem_conv1/BatchNorm/moving_mean/Initializer/zeros:0
ї
5PyramidFusedNet/fem_conv1/BatchNorm/moving_variance:0:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/Assign:PyramidFusedNet/fem_conv1/BatchNorm/moving_variance/read:02FPyramidFusedNet/fem_conv1/BatchNorm/moving_variance/Initializer/ones:0
у
-PyramidFusedNet/fem_conv2/depthwise_weights:02PyramidFusedNet/fem_conv2/depthwise_weights/Assign2PyramidFusedNet/fem_conv2/depthwise_weights/read:02HPyramidFusedNet/fem_conv2/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv2/pointwise_weights:02PyramidFusedNet/fem_conv2/pointwise_weights/Assign2PyramidFusedNet/fem_conv2/pointwise_weights/read:02HPyramidFusedNet/fem_conv2/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv2/BatchNorm/gamma:00PyramidFusedNet/fem_conv2/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv2/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv2/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv2/BatchNorm/beta:0/PyramidFusedNet/fem_conv2/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv2/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv2/BatchNorm/beta/Initializer/zeros:08
ш
1PyramidFusedNet/fem_conv2/BatchNorm/moving_mean:06PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/Assign6PyramidFusedNet/fem_conv2/BatchNorm/moving_mean/read:02CPyramidFusedNet/fem_conv2/BatchNorm/moving_mean/Initializer/zeros:0
ї
5PyramidFusedNet/fem_conv2/BatchNorm/moving_variance:0:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/Assign:PyramidFusedNet/fem_conv2/BatchNorm/moving_variance/read:02FPyramidFusedNet/fem_conv2/BatchNorm/moving_variance/Initializer/ones:0
у
-PyramidFusedNet/fem_conv3/depthwise_weights:02PyramidFusedNet/fem_conv3/depthwise_weights/Assign2PyramidFusedNet/fem_conv3/depthwise_weights/read:02HPyramidFusedNet/fem_conv3/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv3/pointwise_weights:02PyramidFusedNet/fem_conv3/pointwise_weights/Assign2PyramidFusedNet/fem_conv3/pointwise_weights/read:02HPyramidFusedNet/fem_conv3/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv3/BatchNorm/gamma:00PyramidFusedNet/fem_conv3/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv3/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv3/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv3/BatchNorm/beta:0/PyramidFusedNet/fem_conv3/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv3/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv3/BatchNorm/beta/Initializer/zeros:08
ш
1PyramidFusedNet/fem_conv3/BatchNorm/moving_mean:06PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/Assign6PyramidFusedNet/fem_conv3/BatchNorm/moving_mean/read:02CPyramidFusedNet/fem_conv3/BatchNorm/moving_mean/Initializer/zeros:0
ї
5PyramidFusedNet/fem_conv3/BatchNorm/moving_variance:0:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/Assign:PyramidFusedNet/fem_conv3/BatchNorm/moving_variance/read:02FPyramidFusedNet/fem_conv3/BatchNorm/moving_variance/Initializer/ones:0
у
-PyramidFusedNet/fem_conv4/depthwise_weights:02PyramidFusedNet/fem_conv4/depthwise_weights/Assign2PyramidFusedNet/fem_conv4/depthwise_weights/read:02HPyramidFusedNet/fem_conv4/depthwise_weights/Initializer/random_uniform:08
у
-PyramidFusedNet/fem_conv4/pointwise_weights:02PyramidFusedNet/fem_conv4/pointwise_weights/Assign2PyramidFusedNet/fem_conv4/pointwise_weights/read:02HPyramidFusedNet/fem_conv4/pointwise_weights/Initializer/random_uniform:08
б
+PyramidFusedNet/fem_conv4/BatchNorm/gamma:00PyramidFusedNet/fem_conv4/BatchNorm/gamma/Assign0PyramidFusedNet/fem_conv4/BatchNorm/gamma/read:02<PyramidFusedNet/fem_conv4/BatchNorm/gamma/Initializer/ones:08
Ю
*PyramidFusedNet/fem_conv4/BatchNorm/beta:0/PyramidFusedNet/fem_conv4/BatchNorm/beta/Assign/PyramidFusedNet/fem_conv4/BatchNorm/beta/read:02<PyramidFusedNet/fem_conv4/BatchNorm/beta/Initializer/zeros:08
ш
1PyramidFusedNet/fem_conv4/BatchNorm/moving_mean:06PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/Assign6PyramidFusedNet/fem_conv4/BatchNorm/moving_mean/read:02CPyramidFusedNet/fem_conv4/BatchNorm/moving_mean/Initializer/zeros:0
ї
5PyramidFusedNet/fem_conv4/BatchNorm/moving_variance:0:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/Assign:PyramidFusedNet/fem_conv4/BatchNorm/moving_variance/read:02FPyramidFusedNet/fem_conv4/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem0_log_conv0/depthwise_weights:07PyramidFusedNet/dem0_log_conv0/depthwise_weights/Assign7PyramidFusedNet/dem0_log_conv0/depthwise_weights/read:02MPyramidFusedNet/dem0_log_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_log_conv0/pointwise_weights:07PyramidFusedNet/dem0_log_conv0/pointwise_weights/Assign7PyramidFusedNet/dem0_log_conv0/pointwise_weights/read:02MPyramidFusedNet/dem0_log_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma:05PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem0_log_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_log_conv0/BatchNorm/beta:04PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem0_log_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem0_log_conv0/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean:0;PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem0_log_conv0/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance:0?PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem0_log_conv0/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem0_log_conv1/depthwise_weights:07PyramidFusedNet/dem0_log_conv1/depthwise_weights/Assign7PyramidFusedNet/dem0_log_conv1/depthwise_weights/read:02MPyramidFusedNet/dem0_log_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_log_conv1/pointwise_weights:07PyramidFusedNet/dem0_log_conv1/pointwise_weights/Assign7PyramidFusedNet/dem0_log_conv1/pointwise_weights/read:02MPyramidFusedNet/dem0_log_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma:05PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem0_log_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_log_conv1/BatchNorm/beta:04PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem0_log_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem0_log_conv1/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean:0;PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem0_log_conv1/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance:0?PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem0_log_conv1/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem0_reg_conv0/depthwise_weights:07PyramidFusedNet/dem0_reg_conv0/depthwise_weights/Assign7PyramidFusedNet/dem0_reg_conv0/depthwise_weights/read:02MPyramidFusedNet/dem0_reg_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_reg_conv0/pointwise_weights:07PyramidFusedNet/dem0_reg_conv0/pointwise_weights/Assign7PyramidFusedNet/dem0_reg_conv0/pointwise_weights/read:02MPyramidFusedNet/dem0_reg_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma:05PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem0_reg_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta:04PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem0_reg_conv0/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean:0;PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance:0?PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem0_reg_conv0/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem0_reg_conv1/depthwise_weights:07PyramidFusedNet/dem0_reg_conv1/depthwise_weights/Assign7PyramidFusedNet/dem0_reg_conv1/depthwise_weights/read:02MPyramidFusedNet/dem0_reg_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem0_reg_conv1/pointwise_weights:07PyramidFusedNet/dem0_reg_conv1/pointwise_weights/Assign7PyramidFusedNet/dem0_reg_conv1/pointwise_weights/read:02MPyramidFusedNet/dem0_reg_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma:05PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem0_reg_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta:04PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem0_reg_conv1/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean:0;PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance:0?PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem0_reg_conv1/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem1_log_conv0/depthwise_weights:07PyramidFusedNet/dem1_log_conv0/depthwise_weights/Assign7PyramidFusedNet/dem1_log_conv0/depthwise_weights/read:02MPyramidFusedNet/dem1_log_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_log_conv0/pointwise_weights:07PyramidFusedNet/dem1_log_conv0/pointwise_weights/Assign7PyramidFusedNet/dem1_log_conv0/pointwise_weights/read:02MPyramidFusedNet/dem1_log_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma:05PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem1_log_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_log_conv0/BatchNorm/beta:04PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem1_log_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem1_log_conv0/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean:0;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem1_log_conv0/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance:0?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem1_log_conv0/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem1_log_conv1/depthwise_weights:07PyramidFusedNet/dem1_log_conv1/depthwise_weights/Assign7PyramidFusedNet/dem1_log_conv1/depthwise_weights/read:02MPyramidFusedNet/dem1_log_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_log_conv1/pointwise_weights:07PyramidFusedNet/dem1_log_conv1/pointwise_weights/Assign7PyramidFusedNet/dem1_log_conv1/pointwise_weights/read:02MPyramidFusedNet/dem1_log_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma:05PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem1_log_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_log_conv1/BatchNorm/beta:04PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem1_log_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem1_log_conv1/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean:0;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem1_log_conv1/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance:0?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem1_log_conv1/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem1_reg_conv0/depthwise_weights:07PyramidFusedNet/dem1_reg_conv0/depthwise_weights/Assign7PyramidFusedNet/dem1_reg_conv0/depthwise_weights/read:02MPyramidFusedNet/dem1_reg_conv0/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_reg_conv0/pointwise_weights:07PyramidFusedNet/dem1_reg_conv0/pointwise_weights/Assign7PyramidFusedNet/dem1_reg_conv0/pointwise_weights/read:02MPyramidFusedNet/dem1_reg_conv0/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma:05PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/read:02APyramidFusedNet/dem1_reg_conv0/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta:04PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/Assign4PyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/read:02APyramidFusedNet/dem1_reg_conv0/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean:0;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance:0?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem1_reg_conv0/BatchNorm/moving_variance/Initializer/ones:0
ї
2PyramidFusedNet/dem1_reg_conv1/depthwise_weights:07PyramidFusedNet/dem1_reg_conv1/depthwise_weights/Assign7PyramidFusedNet/dem1_reg_conv1/depthwise_weights/read:02MPyramidFusedNet/dem1_reg_conv1/depthwise_weights/Initializer/random_uniform:08
ї
2PyramidFusedNet/dem1_reg_conv1/pointwise_weights:07PyramidFusedNet/dem1_reg_conv1/pointwise_weights/Assign7PyramidFusedNet/dem1_reg_conv1/pointwise_weights/read:02MPyramidFusedNet/dem1_reg_conv1/pointwise_weights/Initializer/random_uniform:08
х
0PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma:05PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/Assign5PyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/read:02APyramidFusedNet/dem1_reg_conv1/BatchNorm/gamma/Initializer/ones:08
т
/PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta:04PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/Assign4PyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/read:02APyramidFusedNet/dem1_reg_conv1/BatchNorm/beta/Initializer/zeros:08
ќ
6PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean:0;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/Assign;PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/read:02HPyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_mean/Initializer/zeros:0

:PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance:0?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/Assign?PyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/read:02KPyramidFusedNet/dem1_reg_conv1/BatchNorm/moving_variance/Initializer/ones:0*Ы
serving_defaultЗ
@
images6
input_images:0"џџџџџџџџџџџџџџџџџџ)
scores
scores:0џџџџџџџџџ,
bboxes"
boxes:0џџџџџџџџџtensorflow/serving/predict