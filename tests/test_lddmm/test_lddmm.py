template = np.zeros([12]*3, dtype=float)
r = 5
for i in range(template.shape[0]):
    for j in range(template.shape[1]):
        for k in range(template.shape[2]):
            if np.sqrt((i-6)**2 + (j-6)**2 + (k-6)**2) <= r:
                template[i,j,k] = 1

target = np.zeros([18, 18, 12], dtype=float)
a, b, c = 8, 8, 5
for i in range(target.shape[0]):
    for j in range(target.shape[1]):
        for k in range(target.shape[2]):
            if (i-9)**2 / a**2 + (j-9)**2 / b**2 + (k-6)**2 / c**2 <= 1:
                target[i,j,k] = 1

template = (template - np.mean(template)) / np.mean(template)
target = (target - np.mean(target)) / np.mean(target)

# holder = register(
#     template, target, 
#     num_iterations=2,
# )

holder = _Holder(template, target)#, affine=[[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])

deformed_subject = _apply_transform(subject=template, subject_resolution=1, output_resolution=None, deform_to="target", holder=holder)



for i in range(0, len(template), len(template)//5):
    plt.imshow(template[i])
    plt.show()