import argparse
from argparse import ArgumentDefaultsHelpFormatter

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for ds_ft_hybrid_4ct models",
                                     formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--phase", type=str, default="train_step1", help="Phase of the training process (train_step1/train_step2/test)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--inputI_size", type=int, default=96, help="Size of input images")
    parser.add_argument("--inputI_chn", type=int, default=1, help="Number of input image channels")
    parser.add_argument("--outputI_size", type=int, default=96, help="Size of output images")
    parser.add_argument("--output_chn", type=int, default=9, help="Number of output image channels")
    parser.add_argument("--pred_filter", type=str, default="0,1,2,3,4,5,6,7,8", help="filter for pred data")
    parser.add_argument("--rename_map", type=str, default="0, 1, 2, 3, 4, 5, 6, 7, 8", help="Renaming map for input/output files")
    parser.add_argument("--resize_r", type=float, default=0.9, help="Resize ratio for input images")
    parser.add_argument("--traindata_dir", type=str, default="../../../HCM_GD_resize/original", help="Directory for training data")
    parser.add_argument("--chkpoint_dir", type=str, default="../outcome/model/checkpoint", help="Directory for train_step1 checkpoints")
    parser.add_argument("--chkpoint_dir2", type=str, default="../outcome/model/checkpoint2", help="Directory for train_step2 checkpoints")
    parser.add_argument("--chkpoint_dir3", type=str, default="../outcome/model/checkpoint3", help="Directory for train_step3 checkpoints")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 parameter for Adam optimizer")
    parser.add_argument("--epoch", type=int, default=54000, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default="ds_ft_hybrid_4ct.model", help="Name of the model")
    parser.add_argument("--save_intval", type=int, default=2000, help="Interval for saving checkpoints")
    parser.add_argument("--testdata_dir", type=str, default="../../../HCM_GD_resize/test/image", help="Directory for test data")
    parser.add_argument("--labeling_dir", type=str, default="../result-test", help="Directory for labeling data")
    parser.add_argument("--testlabel_dir", type=str, default="../../../HCM_GD_resize/test/label", help="Directory for test labels")
    parser.add_argument("--predlabel_dir", type=str, default="../../../HCM_GD_resize/train/pred_label", help="Directory for pred labels")
    parser.add_argument("--ovlp_ita", type=int, default=4, help="Overlap iteration for training")

    args = parser.parse_args()

    # dictionary list
    param_sections = [
        dict(phase=args.phase,
             batch_size=args.batch_size,
             inputI_size=args.inputI_size,
             inputI_chn=args.inputI_chn,
             outputI_size=args.outputI_size,
             output_chn=args.output_chn,
             pred_filter=args.pred_filter,
             rename_map=args.rename_map,
             resize_r=args.resize_r,
             traindata_dir=args.traindata_dir,
             chkpoint_dir=args.chkpoint_dir,
             chkpoint_dir2=args.chkpoint_dir2,
             chkpoint_dir3=args.chkpoint_dir3,
             learning_rate=args.learning_rate,
             beta1=args.beta1,
             epoch=args.epoch,
             model_name=args.model_name,
             save_intval=args.save_intval,
             testdata_dir=args.testdata_dir,
             labeling_dir=args.labeling_dir,
             testlabel_dir=args.testlabel_dir,
             predlabel_dir=args.predlabel_dir,
             ovlp_ita=args.ovlp_ita)
    ]

    return param_sections
