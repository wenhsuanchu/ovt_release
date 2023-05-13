from ytvostools.ytvos import YTVOS
from ytvostools.ytvoseval import YTVOSeval

GT_PATH = "/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense.json"
#GT_PATH = "/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense_trimmed.json"
#GT_PATH = "/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense_perframe.json"
PRED_PATH = "/projects/katefgroup/datasets/UVO/out_uvo/json/merged.json"
#PRED_PATH = "/projects/katefgroup/datasets/UVO/out_uvo/json_perframe/merged.json"
#PRED_PATH = "/home/wenhsuac/ovt/Detic/third_party/SAM/test.json"

def evaluate(test_annotation_file, user_submission_file):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    output = {}
    uvo_api = YTVOS(test_annotation_file)
    print(user_submission_file)
    #uvo_api = uvo_api.loadRes("/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense_with_label_trimmed.json")
    uvo_det = uvo_api.loadRes(user_submission_file)
    # convert ann in uvo_det to class-agnostic
    for ann in uvo_det.dataset["annotations"]:
        if ann["category_id"] != 1:
            ann["category_id"] = 1

    # start evaluation
    uvo_eval = YTVOSeval(uvo_api, uvo_det, "segm")
    uvo_eval.params.useCats = False
    uvo_eval.params.maxDets = [10, 100, 200]
    uvo_eval.evaluate()
    uvo_eval.accumulate()
    uvo_eval.summarize()

    output["result"] = [
        {
            "UVO frame results": {
                "AR@200": uvo_eval.stats[8],
                "AP": uvo_eval.stats[0],
                "AR.5": uvo_eval.stats[9],
                "AR.75": uvo_eval.stats[10],
                "AR@10": uvo_eval.stats[6],
                "AR@100": uvo_eval.stats[7],
            }
        },
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]
    return output

if __name__ == "__main__":
    output = evaluate(GT_PATH, PRED_PATH)
    print(output["submission_result"])