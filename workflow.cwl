#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: Workflow
label: WSU Placenta Clock Challenge Evaluation
doc: |
  This workflow will run and evaluate a Docker submission to the Placenta Clock Challenge. Metrics returned are:
  RMSE.

requirements:
- class: StepInputExpressionRequirement

inputs:
  adminUploadSynId:
    label: Synapse Folder ID accessible by an admin
    type: string
  submissionId:
    label: Submission ID
    type: int
  submitterUploadSynId:
    label: Synapse Folder ID accessible by the submitter
    type: string
  synapseConfig:
    label: filepath to .synapseConfig file
    type: File
  workflowSynapseId:
    label: Synapse File ID that links to the workflow
    type: string

outputs: {}

steps:

  set_submitter_folder_permissions:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/set_permissions.cwl
    in:
      - id: entityid
        source: "#submitterUploadSynId"
      - id: principalid
        valueFrom: "3503823"
      - id: permissions
        valueFrom: "download"
      - id: synapse_config
        source: "#synapseConfig"
    out: []

  set_admin_folder_permissions:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/set_permissions.cwl
    in:
      - id: entityid
        source: "#adminUploadSynId"
      - id: principalid
        valueFrom: "3503823"
      - id: permissions
        valueFrom: "download"
      - id: synapse_config
        source: "#synapseConfig"
    out: []

  get_docker_submission:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/get_submission.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: synapse_config
        source: "#synapseConfig"
    out:
      - id: filepath
      - id: docker_repository
      - id: docker_digest
      - id: entity_id
      - id: entity_type
      - id: evaluation_id
      - id: results

  get_docker_config:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/get_docker_config.cwl
    in:
      - id: synapse_config
        source: "#synapseConfig"
    out: 
      - id: docker_registry
      - id: docker_authentication

  validate_docker:
    run: steps/check_submission.cwl
    in:
      - id: entity_type
        source: "#get_docker_submission/entity_type"
    out:
      - id: results
      - id: status
      - id: invalid_reasons

  email_docker_validation:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/validate_email.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: synapse_config
        source: "#synapseConfig"
      - id: status
        source: "#validate_docker/status"
      - id: invalid_reasons
        source: "#validate_docker/invalid_reasons"
      - id: errors_only
        default: true
    out: [finished]

  annotate_docker_validation_with_output:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/annotate_submission.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: annotation_values
        source: "#validate_docker/results"
      - id: to_public
        default: true
      - id: force
        default: true
      - id: synapse_config
        source: "#synapseConfig"
    out: [finished]

  check_docker_status:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/check_status.cwl
    in:
      - id: status
        source: "#validate_docker/status"
      - id: previous_annotation_finished
        source: "#annotate_docker_validation_with_output/finished"
      - id: previous_email_finished
        source: "#email_docker_validation/finished"
    out: [finished]

  determine_question:
    run: steps/determine_question.cwl
    in:
      - id: queue
        source: "#get_docker_submission/evaluation_id"
    out:
      - id: task_number
 
  run_docker:
    run: steps/run_docker.cwl
    in:
      - id: docker_repository
        source: "#get_docker_submission/docker_repository"
      - id: docker_digest
        source: "#get_docker_submission/docker_digest"
      - id: submissionid
        source: "#submissionId"
      - id: docker_registry
        source: "#get_docker_config/docker_registry"
      - id: docker_authentication
        source: "#get_docker_config/docker_authentication"
      - id: parentid
        source: "#adminUploadSynId"
      - id: synapse_config
        source: "#synapseConfig"
      - id: store
        default: true
      - id: task_number
        source: "#determine_question/task_number"
      - id: docker_script
        default:
          class: File
          location: "run_docker.py"
    out:
      - id: predictions
      - id: results
      - id: status
      - id: invalid_reasons

  email_docker:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.2/cwl/validate_email.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: synapse_config
        source: "#synapseConfig"
      - id: status
        source: "#run_docker/status"
      - id: invalid_reasons
        source: "#run_docker/invalid_reasons"
      - id: errors_only
        default: true
    out: [finished]

  annotate_docker_run_results:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.2/cwl/annotate_submission.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: annotation_values
        source: "#run_docker/results"
      - id: to_public
        default: true
      - id: force
        default: true
      - id: synapse_config
        source: "#synapseConfig"
      - id: previous_annotation_finished
        source: "#annotate_docker_validation_with_output/finished"
    out: [finished]

  check_docker_run_status:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.2/cwl/check_status.cwl
    in:
      - id: status
        source: "#run_docker/status"
      - id: previous_annotation_finished
        source: "#annotate_docker_run_results/finished"
      - id: previous_email_finished
        source: "#email_docker/finished"
    out: [finished]

  upload_results:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/upload_to_synapse.cwl
    in:
      - id: infile
        source: "#run_docker/predictions"
      - id: parentid
        source: "#adminUploadSynId"
      - id: used_entity
        source: "#get_docker_submission/entity_id"
      - id: executed_entity
        source: "#workflowSynapseId"
      - id: synapse_config
        source: "#synapseConfig"
    out:
      - id: uploaded_fileid
      - id: uploaded_file_version
      - id: results

  annotate_docker_upload_results:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/annotate_submission.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: annotation_values
        source: "#upload_results/results"
      - id: to_public
        default: true
      - id: force
        default: true
      - id: synapse_config
        source: "#synapseConfig"
      - id: previous_annotation_finished
        source: "#annotate_docker_run_results/finished"
    out: [finished]

  get_goldstandard_id:
    run: steps/get_goldstandard_id.cwl
    in:
      - id: task_number
        source: "#determine_question/task_number"
    out:
      - id: synid

  download_goldstandard:
    run: https://raw.githubusercontent.com/Sage-Bionetworks-Workflows/cwl-tool-synapseclient/v1.4/cwl/synapse-get-tool.cwl
    in:
      - id: synapseid
        source: "#get_goldstandard_id/synid"
      - id: synapse_config
        source: "#synapseConfig"
    out:
      - id: filepath
  
  validate:
    run: steps/validate.cwl
    in:
      - id: input_file
        source: "#run_docker/predictions"
      - id: goldstandard
        source: "#download_goldstandard/filepath"
      - id: task_number
        source: "#determine_question/task_number"
    out:
      - id: results
      - id: status
      - id: invalid_reasons
  
  email_validation:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/validate_email.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: synapse_config
        source: "#synapseConfig"
      - id: status
        source: "#validate/status"
      - id: invalid_reasons
        source: "#validate/invalid_reasons"
      - id: errors_only
        default: true
    out: [finished]

  annotate_validation_with_output:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/annotate_submission.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: annotation_values
        source: "#validate/results"
      - id: to_public
        default: true
      - id: force
        default: true
      - id: synapse_config
        source: "#synapseConfig"
      - id: previous_annotation_finished
        source: "#annotate_docker_upload_results/finished"
    out: [finished]

  check_status:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/check_status.cwl
    in:
      - id: status
        source: "#validate/status"
      - id: previous_annotation_finished
        source: "#annotate_validation_with_output/finished"
      - id: previous_email_finished
        source: "#email_validation/finished"
    out: [finished]

  score:
    run: steps/score.cwl
    in:
      - id: input_file
        source: "#run_docker/predictions"
      - id: goldstandard
        source: "#download_goldstandard/filepath"
      - id: task_number
        source: "#determine_question/task_number"
      - id: check_validation_finished 
        source: "#check_status/finished"
    out:
      - id: results
      - id: true_scores
      
  email_score:
    run: steps/send_email.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: synapse_config
        source: "#synapseConfig"
      - id: results
        source: "#score/results"
      # - id: private_annotations
      #   default: []
    out: []

  annotate_submission_with_output:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/annotate_submission.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: annotation_values
        source: "#score/results"
      - id: to_public
        default: true
      - id: force
        default: true
      - id: synapse_config
        source: "#synapseConfig"
      - id: previous_annotation_finished
        source: "#annotate_validation_with_output/finished"
    out: [finished]

  annotate_submission_with_hidden_output:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/annotate_submission.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: annotation_values
        source: "#score/true_scores"
      - id: to_public
        default: false
      - id: force
        default: true
      - id: synapse_config
        source: "#synapseConfig"
      - id: previous_annotation_finished
        source: "#annotate_submission_with_output/finished"
    out: [finished]
 
