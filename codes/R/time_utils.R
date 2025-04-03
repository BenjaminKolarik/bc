append_execution_time <- function(time_second, method_name = "LR", excel_file = "output/execution_times/execution_times.xlsx", computer_name) {
  library(openxlsx)

 dir.create(dirname(excel_file), recursive = TRUE, showWarnings = FALSE)
  time_data <- data.frame(
    Method = method_name,
    Computer = computer_name,
    Execution_time = as.numeric(time_second),
    Timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    Language = 'R'
  )

  if(file.exists(excel_file)) {
    wb <- loadWorkbook(excel_file)
        if (method_name %in% names(wb)){
          existing_data <- read.xlsx(wb, sheet = method_name)
          updated_data <- rbind(existing_data, time_data)
          writeData(wb, sheet = method_name, updated_data, startRow = 1)
        } else {
            addWorksheet(wb, method_name)
            writeData(wb, sheet = method_name, time_data, startRow = 1)
        }
  } else{
    wb <- createWorkbook()
    addWorksheet(wb, method_name)
    writeData(wb, sheet = method_name, time_data, startRow = 1)
  }
    saveWorkbook(wb, excel_file, overwrite = TRUE)

  print(paste("Execution time data appended to", excel_file, "in sheet", method_name))
}
