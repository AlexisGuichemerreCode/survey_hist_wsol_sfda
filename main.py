import datetime as dt
import sys
from copy import deepcopy

from dlib.process.parseit import parse_input

from dlib.process.instantiators import get_model
from dlib.utils.tools import log_device
from dlib.utils.tools import bye

from dlib.configure import constants
from dlib.learning.train_wsol import Trainer
from dlib.learning.train_sfuda_sdda_wsol import TrainerSdda
from dlib.process.instantiators import get_pretrainde_classifier
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc

import dlib.dllogger as DLLogger


def main():
    args, args_dict = parse_input(eval=False)
    log_device(args)

    model, model_src = get_model(args)

    model.cuda(args.c_cudaid)
    if model_src:
        model_src.cuda(args.c_cudaid)

    best_state_dict = deepcopy(model.state_dict())

    inter_classifier = None
    if args.task in [constants.F_CL, constants.NEGEV]:
        inter_classifier = get_pretrainde_classifier(args)
        inter_classifier.cuda(args.c_cudaid)

    if args.sf_uda and args.sdda:
        main_trainer = TrainerSdda

    else:
        main_trainer = Trainer

    trainer = main_trainer(args=args,
                           model=model,
                           classifier=inter_classifier,
                           model_src=model_src
                           )

    DLLogger.log(fmsg("Start epoch 0 ..."))

    trainer.evaluate(epoch=0, split=constants.VALIDSET)
    trainer.model_selection(epoch=0)

    trainer.print_performances()
    trainer.report(epoch=0, split=constants.VALIDSET)

    DLLogger.log(fmsg("Epoch 0 done."))

    for epoch in range(1, trainer.args.max_epochs + 1, 1):

        DLLogger.log(fmsg(f"Start epoch {epoch} ..."))

        train_performance = trainer.train(
            split=constants.TRAINSET, epoch=epoch)
        trainer.evaluate(epoch, split=constants.VALIDSET)
        trainer.model_selection(epoch=epoch)

        trainer.report_train(train_performance, epoch)
        trainer.print_performances()
        trainer.report(epoch, split=constants.VALIDSET)
        DLLogger.log(fmsg(("Epoch {} done.".format(epoch))))

        trainer.adjust_learning_rate()
        DLLogger.flush()

    trainer.save_checkpoints()

    trainer.save_best_epoch()
    trainer.capture_perf_meters()

    DLLogger.log(fmsg("Final epoch evaluation on test set ..."))

    if args.task != constants.SEG:
        chpts = [constants.BEST_CL]

        if args.localization_avail:
            chpts = [constants.BEST_LOC] + chpts
    else:
        chpts = [constants.BEST_LOC]

    use_argmax = False

    for eval_checkpoint_type in chpts:
        t0 = dt.datetime.now()

        if eval_checkpoint_type == constants.BEST_LOC:
            epoch = trainer.args.best_loc_epoch
        elif eval_checkpoint_type == constants.BEST_CL:
            epoch = trainer.args.best_cl_epoch
        else:
            raise NotImplementedError

        DLLogger.log(
            fmsg('EVAL TEST SET. CHECKPOINT: {}. ARGMAX: {}'.format(
                eval_checkpoint_type, use_argmax)))

        trainer.load_checkpoint(checkpoint_type=eval_checkpoint_type)

        trainer.evaluate(epoch, split=constants.TESTSET,
                         checkpoint_type=eval_checkpoint_type,
                         fcam_argmax=use_argmax)

        trainer.print_performances(checkpoint_type=eval_checkpoint_type)
        trainer.report(epoch, split=constants.TESTSET,
                       checkpoint_type=eval_checkpoint_type)
        trainer.save_performances(
            epoch=epoch, checkpoint_type=eval_checkpoint_type)

        trainer.switch_perf_meter_to_captured()

        tagargmax = f'Argmax: {use_argmax}'

        DLLogger.log("EVAL time TESTSET - CHECKPOINT {} {}: {}".format(
            eval_checkpoint_type, tagargmax, dt.datetime.now() - t0))
        DLLogger.flush()

    trainer.save_args()
    trainer.plot_perfs_meter()
    bye(trainer.args)


if __name__ == '__main__':
    main()
