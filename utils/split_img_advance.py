import torch


# H: la chieu cao cua tensor, W la chieu rong tensor
# truong hop nay chia duoc so o bang nhau
def get_subboxes_img(H, W, H_oval, W_oval):
    box_list = []
    H_step, W_step = H // H_oval, W // W_oval
    total_oval = H_step * W_step
    
    
    for i in range(total_oval):
        row_i = (i) // W_step # phan du
        col_i = (i) % W_step # phan nguyen
        #print(i+1, row_i, col_i)
        
        x1, y1, x2, y2 = row_i*W_oval, col_i*H_oval, (row_i+1)*W_oval, (col_i+1)*H_oval        
        box_list.append([0, x1, y1, x2, y2])
        
    
    return torch.tensor(box_list, dtype=torch.float32)


# H: la chieu cao cua tensor, W la chieu rong tensor
# So o chia khong deu nhau
def get_subboxes_img_full(H, W, H_oval, W_oval):
    box_list = []
    H_step, W_step = H // H_oval, W // W_oval
        
    H_residual, W_residual = H % H_oval, W % W_oval
    
    if H_residual > 0:
        row_max = H_step + 1
    else:
        row_max = H_step
        
    if W_residual > 0:
        col_max = W_step + 1
    else:
        col_max = W_step
        
    total_oval = row_max * col_max
    
    for i in range(total_oval):
        row_i = (i) // col_max # phan du
        col_i = (i) % col_max # phan nguyen
        x1, y1 = row_i*W_oval, col_i*H_oval
        
        if (W_residual > 0) & (col_i == W_step):
            y2 = W
        else:
            y2 = (col_i+1)*H_oval
        
        if (H_residual > 0) & (row_i == H_step):
            x2 = H
        else:
            x2 = (row_i+1)*W_oval
                
        box_list.append([0, x1, y1, x2, y2])
    
    return torch.tensor(box_list, dtype=torch.float32)

H_oval, W_oval = 512, 512
H, W = 1024, 1536
#H, W = 1024 + 20, 1536 + 20

#print(get_subboxes_img(H, W, H_oval, W_oval))
print(get_subboxes_img_full(H, W, H_oval, W_oval))

