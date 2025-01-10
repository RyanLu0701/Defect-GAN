import argparse 
import torch
from dataset import MyOwnData,ViSAData
from model.model import Generator, Discriminator
from trainer import train
from torch.optim import Adam
from torch.optim import lr_scheduler

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Defect-GAN model")
    

    # Training parameters
    parser.add_argument("--batch_size",      type=int,     default=32,                                   help="Batch size for training")
    parser.add_argument("--epoch",           type=int,     default=2000,                                 help="Number of epochs for training")
    parser.add_argument("--device",          type=str,     default='cuda',                               help="Device to run the training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--gen_update_time", type=int,     default=4,                                    help="Number of discriminator updates before generator update")
    parser.add_argument("--dataset",         type=str,     default='D:\\Ryan\\data_cir\\train',                   help="Path or reference to the training dataloader")
    parser.add_argument("--im_channels",     type=int,     default=3,                                    help="Number of image channels")
    parser.add_argument("--num_worker",      type=int,     default=1,                                    help="Number of image channels")
    parser.add_argument("--size",            type=int,     default=128,                                  help="Image size for training")
    parser.add_argument("--D_lr",            type=int,     default=5e-5,                                 help="learning rate of discriminator")
    parser.add_argument("--G_lr",            type=int,     default=5e-5,                                 help="learning rate of Generator")
    parser.add_argument("--num_classes",     type=int,     default=1,                                    help="How many kind of classes")
     
              
    #Generator parameter              
    parser.add_argument("--num_layers",      type=int,      default=3,                                   help="Number of layers in the model")
    parser.add_argument("--base_channels",   type=int,      default=128,                                 help="Number of base channels for the model")
    parser.add_argument("--repeat_num",      type=int,      default=6,                                   help="Number of times to repeat the base block in the model")
                                        
    #Discriminator parameter                                      
    parser.add_argument("--D_conv_dim",      type=int,      default=128,                                 help="Base convolution dimensions for Discriminator")
    parser.add_argument("--d_c_dim",         type=int,      default=1,                                   help="Number of classes for discriminator (c_dim)")
    parser.add_argument("--d_repeat_num",    type=int,      default=3,                                   help="Number of times to repeat discriminator layers")
                
    # Model parameters                   
    parser.add_argument("--lambda_gp",       type=float,    default=10.0,                                 help="Weight for gradient penalty in discriminator loss")
    parser.add_argument("--lambda_real_cls", type=float,    default=5.0,                                 help="Weight for classification loss in discriminator")
    parser.add_argument("--lambda_fake_cls", type=float,    default=10.0,                                help="Weight for classification loss in generator")
    parser.add_argument("--lambda_rec",      type=float,    default=5.0,                                 help="Weight for reconstruction loss in generator")
    parser.add_argument("--lambda_cyc",      type=float,    default=5.0,                                 help="Weight for spatial cycle-consistency loss in generator")
    parser.add_argument("--lambda_con",      type=float,    default=1.0,                                 help="Weight for region constraint loss in generator")
    args = parser.parse_args()

    vishay_path = "D:\\Ryan\\data_cir\\train"
    train_dataset =  MyOwnData(args.dataset,num_classes=args.num_classes, im_size=args.size, im_channels=3)

    train_Loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    G = Generator(curr_dim=args.im_channels,
                label_nc=args.num_classes,
                num_layers=args.num_layers,
                base_channels=args.base_channels,
                repeat_num=args.repeat_num
                 ).to(args.device)
    D = Discriminator(image_size=args.size,
                      conv_dim=args.D_conv_dim,
                      c_dim=args.d_c_dim,
                      repeat_num=args.d_repeat_num).to(args.device)
    
    G_optimizer = Adam(G.parameters(), lr=args.G_lr ,betas=(0.5,0.999))
    D_optimizer = Adam(D.parameters(), lr=args.D_lr ,betas=(0.5,0.999))

    g_scheduler = lr_scheduler.ReduceLROnPlateau(G_optimizer, mode='min', factor=0.8, patience=10, verbose=False)
    d_scheduler = lr_scheduler.ReduceLROnPlateau(D_optimizer, mode='min', factor=0.8, patience=10, verbose=False)



    train(args=args, dataloader=train_Loader, discriminator=D, generator=G, G_optimizer=G_optimizer, D_optimizer=D_optimizer, g_scheduler=g_scheduler, d_scheduler=d_scheduler)