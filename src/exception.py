import sys
from logger import logging
def error_message_detail(error, error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    if exc_tb is not None:
        try:
            file_name = exc_tb.tbframe.f_code.co_filename
            error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name, exc_tb.tb_lineno, str(error))
            return error_message     
        except AttributeError:
            return f"Error: {str(error)}" # handle rare attribute error.
    else:
        return f"Error: {str(error)}" # handle case where exc_tb is None.                                                                                                   
                                                                                                          

class CustomException(Exception):
    def __init__(self,error_messsage,error_details:sys):
        super().__init__(error_messsage)
        self.error_message = error_message_detail(error_messsage,error_details=error_details)
    
    def __str__(self):
        return self.error_message
    

if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        raise CustomException(e,sys)
        logging.info("Divided by 0")
        