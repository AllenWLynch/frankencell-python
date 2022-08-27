from dynclipy.read import ro as dynro
import os

def run_eval(*, goldstandard, test_dataset, method,
    method_output_path, results_output_path,
    param_string = ''):

    prefix_path = os.path.join(os.path.dirname(__file__), 'methods', method)
    definition_file = os.path.join(prefix_path, 'definition.yaml')
    run_file = os.path.join(prefix_path, 'run')

    os.environ['frankencell_method_definition_path'] = os.path.abspath(definition_file)

    path = os.path.dirname(os.path.abspath(__file__))
    rstring = '''
goldstandard <- dynutils::read_h5("{goldstandard}")
model <- dynutils::read_h5("{test_dataset}")

test_method <- dynwrap::create_ti_method_definition(
    "{definition_file}",
    "{run_file}",
)

model <- dynwrap::infer_trajectory(model, test_method({param_string}), verbose = TRUE,
            give_priors = c('start_id','end_id','dimred'))

model <- dynwrap::add_cell_waypoints(model)

results <- dyneval::calculate_metrics(goldstandard, model,
                        metrics = c("correlation","F1_branches","edge_flip"))

dynutils::write_h5(model, "{method_output_path}")

write.table(t(results), file = "{results_output_path}", 
    sep = "\t", col.names = FALSE, quote = FALSE)

    '''.format(
        path = path,
        run_file = run_file,
        definition_file = definition_file,
        goldstandard = goldstandard,
        test_dataset = test_dataset,
        method_output_path = method_output_path,
        param_string = param_string,
        results_output_path = results_output_path
    )

    dynro.r(rstring)

def main(parser):

    run_eval(
        goldstandard=parser.ground_truth,
        test_dataset=parser.ground_truth,
        method = parser.method,
        method_output_path = parser.out_h5,
        results_output_path=parser.results_file,
        param_string=parser.parameters,
    )


def add_arguments(parser):
    parser.add_argument('method', type = str)
    parser.add_argument('--ground-truth', '-t', type = str, required = True)
    parser.add_argument('--out-h5', '-o', type = str, required = True)
    parser.add_argument('--results-file', '-r', type = str, required = True)
    parser.add_argument('--parameters', '-p', type = str, required = False,
        default = '')